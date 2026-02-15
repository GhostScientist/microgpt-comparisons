// microgpt.m
// The most atomic way to train and inference a GPT in pure Objective-C.
// Ported from Karpathy's microgpt.py — Foundation only.

#import <Foundation/Foundation.h>
#import <math.h>

// =========================================================================
// Autograd engine
// =========================================================================

@interface Value : NSObject {
    @public
    double _data;
    double _grad;
    NSArray<Value *> *_children;
    double _lg[2];
    int _nc;
}
@end

@implementation Value

- (instancetype)initWithData:(double)d children:(NSArray<Value *> *)ch lg0:(double)lg0 lg1:(double)lg1 nc:(int)nc {
    if ((self = [super init])) {
        _data = d; _grad = 0; _children = ch; _lg[0] = lg0; _lg[1] = lg1; _nc = nc;
    }
    return self;
}

- (instancetype)initWithData:(double)d {
    return [self initWithData:d children:@[] lg0:0 lg1:0 nc:0];
}

- (Value *)logValue {
    return [[Value alloc] initWithData:log(_data) children:@[self] lg0:1.0/_data lg1:0 nc:1];
}

- (Value *)expValue {
    double e = exp(_data);
    return [[Value alloc] initWithData:e children:@[self] lg0:e lg1:0 nc:1];
}

- (Value *)reluValue {
    return [[Value alloc] initWithData:fmax(0, _data) children:@[self] lg0:(_data > 0 ? 1.0 : 0.0) lg1:0 nc:1];
}

- (Value *)powValue:(double)n {
    return [[Value alloc] initWithData:pow(_data, n) children:@[self] lg0:n * pow(_data, n - 1) lg1:0 nc:1];
}

- (void)backward {
    NSMutableArray<Value *> *topo = [NSMutableArray array];
    NSMutableSet<Value *> *visited = [NSMutableSet set];
    [self _buildTopo:self into:topo visited:visited];
    _grad = 1.0;
    for (NSInteger i = (NSInteger)topo.count - 1; i >= 0; i--) {
        Value *v = topo[(NSUInteger)i];
        for (int c = 0; c < v->_nc; c++) {
            v->_children[c]->_grad += v->_lg[c] * v->_grad;
        }
    }
}

- (void)_buildTopo:(Value *)v into:(NSMutableArray *)topo visited:(NSMutableSet *)visited {
    if ([visited containsObject:v]) return;
    [visited addObject:v];
    for (Value *ch in v->_children) [self _buildTopo:ch into:topo visited:visited];
    [topo addObject:v];
}

@end

// =========================================================================
// Operator helpers (free functions)
// =========================================================================

static Value *val_add(Value *a, Value *b) {
    return [[Value alloc] initWithData:a->_data + b->_data children:@[a, b] lg0:1 lg1:1 nc:2];
}

static Value *val_mul(Value *a, Value *b) {
    return [[Value alloc] initWithData:a->_data * b->_data children:@[a, b] lg0:b->_data lg1:a->_data nc:2];
}

static Value *val_add_d(Value *a, double c) { return val_add(a, [[Value alloc] initWithData:c]); }
static Value *val_mul_d(Value *a, double c) { return val_mul(a, [[Value alloc] initWithData:c]); }
static Value *val_neg(Value *a) { return val_mul_d(a, -1.0); }
static Value *val_div(Value *a, Value *b) { return val_mul(a, [b powValue:-1]); }
static Value *val_div_d(Value *a, double c) { return val_mul_d(a, 1.0 / c); }

// =========================================================================
// Helpers
// =========================================================================

static double gaussRandom(double std) {
    double u1 = drand48(), u2 = drand48();
    return std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static NSInteger weightedChoice(double *w, NSUInteger n) {
    double total = 0;
    for (NSUInteger i = 0; i < n; i++) total += w[i];
    double r = drand48() * total;
    for (NSUInteger i = 0; i < n; i++) { r -= w[i]; if (r <= 0) return (NSInteger)i; }
    return (NSInteger)(n - 1);
}

static Value *sumValues(Value *__strong *vals, NSUInteger n) {
    Value *s = vals[0];
    for (NSUInteger i = 1; i < n; i++) s = val_add(s, vals[i]);
    return s;
}

static NSArray<NSArray<Value *> *> *makeMatrix(NSInteger nout, NSInteger nin, double std) {
    NSMutableArray *rows = [NSMutableArray arrayWithCapacity:(NSUInteger)nout];
    for (NSInteger i = 0; i < nout; i++) {
        NSMutableArray *row = [NSMutableArray arrayWithCapacity:(NSUInteger)nin];
        for (NSInteger j = 0; j < nin; j++)
            [row addObject:[[Value alloc] initWithData:gaussRandom(std)]];
        [rows addObject:row];
    }
    return rows;
}

// Linear: x[nin] -> out[nout] via w[nout][nin]
static void linearLayer(Value *__strong *out, NSUInteger *outLen,
                         Value *__strong *x, NSUInteger xLen,
                         NSArray<NSArray<Value *> *> *w) {
    NSUInteger nout = w.count;
    *outLen = nout;
    for (NSUInteger i = 0; i < nout; i++) {
        NSArray<Value *> *row = w[i];
        NSUInteger nin = row.count;
        Value *__strong *prods = (Value *__strong *)calloc(nin, sizeof(Value *));
        for (NSUInteger j = 0; j < nin; j++) prods[j] = val_mul(row[j], x[j]);
        out[i] = sumValues(prods, nin);
        for (NSUInteger j = 0; j < nin; j++) prods[j] = nil;
        free(prods);
    }
}

static void softmaxLayer(Value *__strong *out, Value *__strong *logits, NSUInteger n) {
    double mx = logits[0]->_data;
    for (NSUInteger i = 1; i < n; i++) if (logits[i]->_data > mx) mx = logits[i]->_data;
    Value *__strong *exps = (Value *__strong *)calloc(n, sizeof(Value *));
    for (NSUInteger i = 0; i < n; i++) exps[i] = [val_add_d(logits[i], -mx) expValue];
    Value *total = sumValues(exps, n);
    for (NSUInteger i = 0; i < n; i++) out[i] = val_div(exps[i], total);
    for (NSUInteger i = 0; i < n; i++) exps[i] = nil;
    free(exps);
}

static void rmsnormLayer(Value *__strong *out, Value *__strong *x, NSUInteger n) {
    Value *__strong *sq = (Value *__strong *)calloc(n, sizeof(Value *));
    for (NSUInteger i = 0; i < n; i++) sq[i] = val_mul(x[i], x[i]);
    Value *ms = val_div_d(sumValues(sq, n), (double)n);
    Value *scale = [val_add_d(ms, 1e-5) powValue:-0.5];
    for (NSUInteger i = 0; i < n; i++) out[i] = val_mul(x[i], scale);
    for (NSUInteger i = 0; i < n; i++) sq[i] = nil;
    free(sq);
}

static void freeVA(Value *__strong *a, NSUInteger n) {
    if (!a) return;
    for (NSUInteger i = 0; i < n; i++) a[i] = nil;
    free(a);
}

// =========================================================================
// Main
// =========================================================================

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        const NSInteger nEmbd = 16, nHead = 4, nLayer = 1, blockSize = 16, headDim = 4;

        srand48(42);

        // Load dataset
        NSString *inputPath = @"input.txt";
        if (![[NSFileManager defaultManager] fileExistsAtPath:inputPath]) {
            printf("Downloading names dataset...\n");
            NSData *dl = [NSData dataWithContentsOfURL:
                [NSURL URLWithString:@"https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"]];
            [dl writeToFile:inputPath atomically:YES];
        }

        NSString *content = [NSString stringWithContentsOfFile:inputPath encoding:NSUTF8StringEncoding error:nil];
        NSMutableArray<NSString *> *docs = [NSMutableArray array];
        for (NSString *line in [content componentsSeparatedByString:@"\n"]) {
            NSString *t = [line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            if (t.length > 0) [docs addObject:t];
        }

        for (NSInteger i = (NSInteger)docs.count - 1; i >= 1; i--) {
            NSInteger j = (NSInteger)(drand48() * (double)(i + 1));
            [docs exchangeObjectAtIndex:(NSUInteger)i withObjectAtIndex:(NSUInteger)j];
        }
        printf("num docs: %lu\n", (unsigned long)docs.count);

        // Tokenizer
        NSMutableSet<NSNumber *> *charSet = [NSMutableSet set];
        for (NSString *doc in docs)
            for (NSUInteger i = 0; i < doc.length; i++)
                [charSet addObject:@([doc characterAtIndex:i])];

        NSArray<NSNumber *> *sortedChars = [[charSet allObjects] sortedArrayUsingSelector:@selector(compare:)];
        NSUInteger ucharCount = sortedChars.count;
        unichar *uchars = (unichar *)malloc(ucharCount * sizeof(unichar));
        NSMutableDictionary<NSNumber *, NSNumber *> *charToIdx = [NSMutableDictionary dictionary];
        for (NSUInteger i = 0; i < ucharCount; i++) {
            uchars[i] = sortedChars[i].unsignedShortValue;
            charToIdx[sortedChars[i]] = @(i);
        }
        NSInteger BOS = (NSInteger)ucharCount;
        NSInteger vocabSize = (NSInteger)ucharCount + 1;
        printf("vocab size: %ld\n", (long)vocabSize);

        // Model parameters
        double initStd = 0.08;
        NSMutableDictionary<NSString *, NSArray<NSArray<Value *> *> *> *sd = [NSMutableDictionary dictionary];
        sd[@"wte"] = makeMatrix(vocabSize, nEmbd, initStd);
        sd[@"wpe"] = makeMatrix(blockSize, nEmbd, initStd);
        sd[@"lm_head"] = makeMatrix(vocabSize, nEmbd, initStd);
        for (NSInteger li = 0; li < nLayer; li++) {
            NSString *p = [NSString stringWithFormat:@"layer%ld", (long)li];
            sd[[p stringByAppendingString:@".attn_wq"]] = makeMatrix(nEmbd, nEmbd, initStd);
            sd[[p stringByAppendingString:@".attn_wk"]] = makeMatrix(nEmbd, nEmbd, initStd);
            sd[[p stringByAppendingString:@".attn_wv"]] = makeMatrix(nEmbd, nEmbd, initStd);
            sd[[p stringByAppendingString:@".attn_wo"]] = makeMatrix(nEmbd, nEmbd, initStd);
            sd[[p stringByAppendingString:@".mlp_fc1"]] = makeMatrix(4 * nEmbd, nEmbd, initStd);
            sd[[p stringByAppendingString:@".mlp_fc2"]] = makeMatrix(nEmbd, 4 * nEmbd, initStd);
        }

        NSArray<NSString *> *sortedKeys = [[sd allKeys] sortedArrayUsingSelector:@selector(compare:)];
        NSMutableArray<Value *> *paramsArr = [NSMutableArray array];
        for (NSString *key in sortedKeys)
            for (NSArray<Value *> *row in sd[key])
                for (Value *v in row)
                    [paramsArr addObject:v];
        NSUInteger paramCount = paramsArr.count;
        printf("num params: %lu\n", (unsigned long)paramCount);

        Value *__strong *params = (Value *__strong *)calloc(paramCount, sizeof(Value *));
        for (NSUInteger i = 0; i < paramCount; i++) params[i] = paramsArr[i];

        double *mAdam = (double *)calloc(paramCount, sizeof(double));
        double *vAdam = (double *)calloc(paramCount, sizeof(double));

        // Training
        for (NSInteger step = 0; step < 1000; step++) {
            @autoreleasepool {
                NSString *doc = docs[(NSUInteger)(step % (NSInteger)docs.count)];
                NSUInteger docLen = doc.length;
                NSInteger *tokens = (NSInteger *)malloc((docLen + 2) * sizeof(NSInteger));
                tokens[0] = BOS;
                for (NSUInteger ci = 0; ci < docLen; ci++)
                    tokens[ci + 1] = charToIdx[@([doc characterAtIndex:ci])].integerValue;
                tokens[docLen + 1] = BOS;

                NSInteger n = blockSize < (NSInteger)(docLen + 1) ? blockSize : (NSInteger)(docLen + 1);

                // KV caches
                NSMutableArray<NSMutableArray<NSArray<Value *> *> *> *keysC = [NSMutableArray array];
                NSMutableArray<NSMutableArray<NSArray<Value *> *> *> *valsC = [NSMutableArray array];
                for (NSInteger li = 0; li < nLayer; li++) {
                    [keysC addObject:[NSMutableArray array]];
                    [valsC addObject:[NSMutableArray array]];
                }

                Value *__strong *losses = (Value *__strong *)calloc((NSUInteger)n, sizeof(Value *));

                for (NSInteger posId = 0; posId < n; posId++) {
                    NSInteger tokenId = tokens[posId];
                    NSInteger targetId = tokens[posId + 1];

                    // GPT forward
                    NSArray<Value *> *tokEmb = sd[@"wte"][(NSUInteger)tokenId];
                    NSArray<Value *> *posEmb = sd[@"wpe"][(NSUInteger)posId];
                    Value *__strong *x = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                    for (NSInteger e = 0; e < nEmbd; e++) x[e] = val_add(tokEmb[(NSUInteger)e], posEmb[(NSUInteger)e]);

                    Value *__strong *xn = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                    rmsnormLayer(xn, x, (NSUInteger)nEmbd);
                    freeVA(x, (NSUInteger)nEmbd); x = xn;

                    for (NSInteger li = 0; li < nLayer; li++) {
                        NSString *pfx = [NSString stringWithFormat:@"layer%ld", (long)li];
                        Value *__strong *xRes = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        for (NSInteger e = 0; e < nEmbd; e++) xRes[e] = x[e];

                        xn = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        rmsnormLayer(xn, x, (NSUInteger)nEmbd);
                        freeVA(x, (NSUInteger)nEmbd); x = xn;

                        NSUInteger qL, kL, vL;
                        Value *__strong *q = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        Value *__strong *k = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        Value *__strong *v = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        linearLayer(q, &qL, x, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".attn_wq"]]);
                        linearLayer(k, &kL, x, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".attn_wk"]]);
                        linearLayer(v, &vL, x, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".attn_wv"]]);

                        // Store k, v as NSArray for KV cache
                        NSMutableArray<Value *> *kArr = [NSMutableArray arrayWithCapacity:(NSUInteger)nEmbd];
                        NSMutableArray<Value *> *vArr = [NSMutableArray arrayWithCapacity:(NSUInteger)nEmbd];
                        for (NSInteger e = 0; e < nEmbd; e++) { [kArr addObject:k[e]]; [vArr addObject:v[e]]; }
                        [keysC[(NSUInteger)li] addObject:kArr];
                        [valsC[(NSUInteger)li] addObject:vArr];

                        Value *__strong *xAttn = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        NSUInteger ai = 0;
                        double sf = sqrt((double)headDim);

                        for (NSInteger h = 0; h < nHead; h++) {
                            NSUInteger hs = (NSUInteger)(h * headDim);
                            NSUInteger numC = keysC[(NSUInteger)li].count;

                            Value *__strong *al = (Value *__strong *)calloc(numC, sizeof(Value *));
                            for (NSUInteger t = 0; t < numC; t++) {
                                Value *__strong *pr = (Value *__strong *)calloc((NSUInteger)headDim, sizeof(Value *));
                                for (NSInteger d = 0; d < headDim; d++)
                                    pr[d] = val_mul(q[hs + (NSUInteger)d], keysC[(NSUInteger)li][t][hs + (NSUInteger)d]);
                                al[t] = val_div_d(sumValues(pr, (NSUInteger)headDim), sf);
                                freeVA(pr, (NSUInteger)headDim);
                            }
                            Value *__strong *aw = (Value *__strong *)calloc(numC, sizeof(Value *));
                            softmaxLayer(aw, al, numC);
                            freeVA(al, numC);

                            for (NSInteger d = 0; d < headDim; d++) {
                                Value *__strong *terms = (Value *__strong *)calloc(numC, sizeof(Value *));
                                for (NSUInteger t = 0; t < numC; t++)
                                    terms[t] = val_mul(aw[t], valsC[(NSUInteger)li][t][hs + (NSUInteger)d]);
                                xAttn[ai++] = sumValues(terms, numC);
                                freeVA(terms, numC);
                            }
                            freeVA(aw, numC);
                        }
                        freeVA(q, (NSUInteger)nEmbd);

                        NSUInteger oL;
                        freeVA(x, (NSUInteger)nEmbd);
                        x = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        linearLayer(x, &oL, xAttn, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".attn_wo"]]);
                        freeVA(xAttn, (NSUInteger)nEmbd);

                        for (NSInteger e = 0; e < nEmbd; e++) x[e] = val_add(x[e], xRes[e]);
                        freeVA(xRes, (NSUInteger)nEmbd);

                        // MLP
                        Value *__strong *xRes2 = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        for (NSInteger e = 0; e < nEmbd; e++) xRes2[e] = x[e];

                        xn = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        rmsnormLayer(xn, x, (NSUInteger)nEmbd);
                        freeVA(x, (NSUInteger)nEmbd); x = xn;

                        NSUInteger f1L;
                        Value *__strong *h1 = (Value *__strong *)calloc((NSUInteger)(4 * nEmbd), sizeof(Value *));
                        linearLayer(h1, &f1L, x, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".mlp_fc1"]]);
                        freeVA(x, (NSUInteger)nEmbd);
                        for (NSUInteger ri = 0; ri < f1L; ri++) h1[ri] = [h1[ri] reluValue];

                        NSUInteger f2L;
                        x = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        linearLayer(x, &f2L, h1, f1L, sd[[pfx stringByAppendingString:@".mlp_fc2"]]);
                        freeVA(h1, f1L);

                        for (NSInteger e = 0; e < nEmbd; e++) x[e] = val_add(x[e], xRes2[e]);
                        freeVA(xRes2, (NSUInteger)nEmbd);

                        freeVA(k, (NSUInteger)nEmbd);
                        freeVA(v, (NSUInteger)nEmbd);
                    }

                    NSUInteger lgL;
                    Value *__strong *logits = (Value *__strong *)calloc((NSUInteger)vocabSize, sizeof(Value *));
                    linearLayer(logits, &lgL, x, (NSUInteger)nEmbd, sd[@"lm_head"]);
                    freeVA(x, (NSUInteger)nEmbd);

                    Value *__strong *probs = (Value *__strong *)calloc(lgL, sizeof(Value *));
                    softmaxLayer(probs, logits, lgL);
                    freeVA(logits, lgL);

                    losses[posId] = val_neg([probs[(NSUInteger)targetId] logValue]);
                    freeVA(probs, lgL);
                }

                Value *loss = val_div_d(sumValues(losses, (NSUInteger)n), (double)n);
                freeVA(losses, (NSUInteger)n);
                [loss backward];

                double lrT = 0.01 * (1.0 - (double)step / 1000.0);
                for (NSUInteger i = 0; i < paramCount; i++) {
                    double g = params[i]->_grad;
                    mAdam[i] = 0.85 * mAdam[i] + 0.15 * g;
                    vAdam[i] = 0.99 * vAdam[i] + 0.01 * g * g;
                    double mH = mAdam[i] / (1.0 - pow(0.85, (double)(step + 1)));
                    double vH = vAdam[i] / (1.0 - pow(0.99, (double)(step + 1)));
                    params[i]->_data -= lrT * mH / (sqrt(vH) + 1e-8);
                    params[i]->_grad = 0;
                }

                printf("step %4ld / %4d | loss %.4f\n", (long)(step + 1), 1000, loss->_data);
                free(tokens);
            }
        }

        // Inference
        printf("\n--- inference (new, hallucinated names) ---\n");
        for (NSInteger si = 0; si < 20; si++) {
            @autoreleasepool {
                NSMutableArray<NSMutableArray<NSArray<Value *> *> *> *keysC = [NSMutableArray array];
                NSMutableArray<NSMutableArray<NSArray<Value *> *> *> *valsC = [NSMutableArray array];
                for (NSInteger li = 0; li < nLayer; li++) {
                    [keysC addObject:[NSMutableArray array]];
                    [valsC addObject:[NSMutableArray array]];
                }

                NSInteger tokenId = BOS;
                NSMutableString *sample = [NSMutableString string];

                for (NSInteger posId = 0; posId < blockSize; posId++) {
                    NSArray<Value *> *tokEmb = sd[@"wte"][(NSUInteger)tokenId];
                    NSArray<Value *> *posEmb = sd[@"wpe"][(NSUInteger)posId];
                    Value *__strong *x = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                    for (NSInteger e = 0; e < nEmbd; e++) x[e] = val_add(tokEmb[(NSUInteger)e], posEmb[(NSUInteger)e]);

                    Value *__strong *xn = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                    rmsnormLayer(xn, x, (NSUInteger)nEmbd);
                    freeVA(x, (NSUInteger)nEmbd); x = xn;

                    for (NSInteger li = 0; li < nLayer; li++) {
                        NSString *pfx = [NSString stringWithFormat:@"layer%ld", (long)li];
                        Value *__strong *xRes = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        for (NSInteger e = 0; e < nEmbd; e++) xRes[e] = x[e];

                        xn = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        rmsnormLayer(xn, x, (NSUInteger)nEmbd);
                        freeVA(x, (NSUInteger)nEmbd); x = xn;

                        NSUInteger qL, kL, vL;
                        Value *__strong *q = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        Value *__strong *k = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        Value *__strong *v = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        linearLayer(q, &qL, x, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".attn_wq"]]);
                        linearLayer(k, &kL, x, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".attn_wk"]]);
                        linearLayer(v, &vL, x, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".attn_wv"]]);

                        NSMutableArray<Value *> *kArr = [NSMutableArray arrayWithCapacity:(NSUInteger)nEmbd];
                        NSMutableArray<Value *> *vArr = [NSMutableArray arrayWithCapacity:(NSUInteger)nEmbd];
                        for (NSInteger e = 0; e < nEmbd; e++) { [kArr addObject:k[e]]; [vArr addObject:v[e]]; }
                        [keysC[(NSUInteger)li] addObject:kArr];
                        [valsC[(NSUInteger)li] addObject:vArr];

                        Value *__strong *xAttn = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        NSUInteger ai = 0;
                        double sf = sqrt((double)headDim);
                        for (NSInteger h = 0; h < nHead; h++) {
                            NSUInteger hs = (NSUInteger)(h * headDim);
                            NSUInteger numC = keysC[(NSUInteger)li].count;
                            Value *__strong *al = (Value *__strong *)calloc(numC, sizeof(Value *));
                            for (NSUInteger t = 0; t < numC; t++) {
                                Value *__strong *pr = (Value *__strong *)calloc((NSUInteger)headDim, sizeof(Value *));
                                for (NSInteger d = 0; d < headDim; d++)
                                    pr[d] = val_mul(q[hs + (NSUInteger)d], keysC[(NSUInteger)li][t][hs + (NSUInteger)d]);
                                al[t] = val_div_d(sumValues(pr, (NSUInteger)headDim), sf);
                                freeVA(pr, (NSUInteger)headDim);
                            }
                            Value *__strong *aw = (Value *__strong *)calloc(numC, sizeof(Value *));
                            softmaxLayer(aw, al, numC);
                            freeVA(al, numC);
                            for (NSInteger d = 0; d < headDim; d++) {
                                Value *__strong *terms = (Value *__strong *)calloc(numC, sizeof(Value *));
                                for (NSUInteger t = 0; t < numC; t++)
                                    terms[t] = val_mul(aw[t], valsC[(NSUInteger)li][t][hs + (NSUInteger)d]);
                                xAttn[ai++] = sumValues(terms, numC);
                                freeVA(terms, numC);
                            }
                            freeVA(aw, numC);
                        }
                        freeVA(q, (NSUInteger)nEmbd);
                        NSUInteger oL;
                        freeVA(x, (NSUInteger)nEmbd);
                        x = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        linearLayer(x, &oL, xAttn, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".attn_wo"]]);
                        freeVA(xAttn, (NSUInteger)nEmbd);
                        for (NSInteger e = 0; e < nEmbd; e++) x[e] = val_add(x[e], xRes[e]);
                        freeVA(xRes, (NSUInteger)nEmbd);

                        Value *__strong *xRes2 = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        for (NSInteger e = 0; e < nEmbd; e++) xRes2[e] = x[e];
                        xn = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        rmsnormLayer(xn, x, (NSUInteger)nEmbd);
                        freeVA(x, (NSUInteger)nEmbd); x = xn;
                        NSUInteger f1L;
                        Value *__strong *h1 = (Value *__strong *)calloc((NSUInteger)(4 * nEmbd), sizeof(Value *));
                        linearLayer(h1, &f1L, x, (NSUInteger)nEmbd, sd[[pfx stringByAppendingString:@".mlp_fc1"]]);
                        freeVA(x, (NSUInteger)nEmbd);
                        for (NSUInteger ri = 0; ri < f1L; ri++) h1[ri] = [h1[ri] reluValue];
                        NSUInteger f2L;
                        x = (Value *__strong *)calloc((NSUInteger)nEmbd, sizeof(Value *));
                        linearLayer(x, &f2L, h1, f1L, sd[[pfx stringByAppendingString:@".mlp_fc2"]]);
                        freeVA(h1, f1L);
                        for (NSInteger e = 0; e < nEmbd; e++) x[e] = val_add(x[e], xRes2[e]);
                        freeVA(xRes2, (NSUInteger)nEmbd);
                        freeVA(k, (NSUInteger)nEmbd);
                        freeVA(v, (NSUInteger)nEmbd);
                    }

                    NSUInteger lgL;
                    Value *__strong *logits = (Value *__strong *)calloc((NSUInteger)vocabSize, sizeof(Value *));
                    linearLayer(logits, &lgL, x, (NSUInteger)nEmbd, sd[@"lm_head"]);
                    freeVA(x, (NSUInteger)nEmbd);

                    Value *__strong *scaled = (Value *__strong *)calloc(lgL, sizeof(Value *));
                    for (NSUInteger i = 0; i < lgL; i++) scaled[i] = val_div_d(logits[i], 0.5);
                    freeVA(logits, lgL);

                    Value *__strong *probs = (Value *__strong *)calloc(lgL, sizeof(Value *));
                    softmaxLayer(probs, scaled, lgL);
                    freeVA(scaled, lgL);

                    double *pd = (double *)malloc(lgL * sizeof(double));
                    for (NSUInteger i = 0; i < lgL; i++) pd[i] = probs[i]->_data;
                    freeVA(probs, lgL);

                    tokenId = weightedChoice(pd, lgL);
                    free(pd);
                    if (tokenId == BOS) break;
                    [sample appendFormat:@"%C", uchars[(NSUInteger)tokenId]];
                }
                printf("sample %2ld: %s\n", (long)(si + 1), [sample UTF8String]);
            }
        }

        free(mAdam); free(vAdam);
        for (NSUInteger i = 0; i < paramCount; i++) params[i] = nil;
        free(params);
        free(uchars);
    }
    return 0;
}
