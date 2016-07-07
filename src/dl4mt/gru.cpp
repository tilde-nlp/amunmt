#include "gru.h"
#include "simd_math_prims.h"
#include "quant/operators.h"

void gElementwiseOps(data_t* out,
                    const data_t* state,
                    /*****************/
                    const data_t* ruh,
                    const data_t* t,
                    const data_t* b,
                    const data_t* bx1,
                    const data_t* bx2,
                    float minTotal, float maxTotal,
                    /*****************/
                    size_t rows, size_t cols) {
  
  data_t zeroTotal = FloatToQuantized<data_t>(0.0f, minTotal, maxTotal);
  data_t oneTotal = FloatToQuantized<data_t>(1.0f, minTotal, maxTotal);
  
  data_t zero = FloatToQuantized<data_t>(0.0f, -1, 1);
  data_t one = FloatToQuantized<data_t>(1.0f, -1, 1);
  
  data_t logitTotal[256];
  data_t logit[256];
  data_t tanh[256];
  for(int i = 0; i < 256; ++i) {
    float f = QuantizedToFloat<data_t>((data_t)i, minTotal, maxTotal);
    logitTotal[i] = FloatToQuantized<data_t>(logitapprox(f), minTotal, maxTotal);
    logit[i] = FloatToQuantized<data_t>(logitapprox(f), -1.0, 1.0);
    tanh[i] = FloatToQuantized<data_t>(tanhapprox(f), -1.0, 1.0);
  }
  
  for(int j = 0; j < rows; ++j) {
    const data_t* rowRuh = ruh + j * cols * 3;
    const data_t* rowH = rowRuh + 2 * cols;
    
    const data_t* rowT1 = t + j * cols * 3;
    const data_t* rowT2 = rowT1 + 2 * cols;
    
    const data_t* rowState = state + j * cols;
    data_t* rowOut = out + j * cols;
    
    for(int i = 0; i < cols; ++i) {
      int k = i + cols;
      
      data_t mul = qmul(zeroTotal, oneTotal,
                        logitTotal[qsum(zeroTotal, rowRuh[i], rowT1[i], b[i])],
                        qsum(zeroTotal, rowT2[i], bx2[i]));
      data_t h = tanh[qsum(zeroTotal, rowH[i], bx1[i], mul)];
      data_t u = logit[qsum(zeroTotal, rowRuh[k], rowT1[k], b[k])];
      
      rowOut[i] = qsum(zero, 
                    qmul(zero, one, qsub(zero, one, u), h),
                    qmul(zero, one, u, rowState[i]));
    }
  }
}
