#include <chrono>

#define GEMMLOWP_SSE4_64

#include "god.h"
#include "encoder_decoder/encoder_decoder.h"
#include "decoder/sentence.h"
#include "dl4mt.h"

int main(int argc, char* argv[]) {
  //God::Init(argc, argv);

  using namespace mblas;
  
  gemmlowp::GemmContext context;
  context.set_max_num_threads(16);

  //QMatrix a1(12, 500);
  //QMatrix b1(500, 80000);  
  //Matrix c1;
  //          
  //std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
  //for(int i = 0; i < 100; ++i)
  //  QProd(context, c1, a1, b1);
  //std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
  //std::chrono::duration<double> fp_s1 = end1 - start1;
  //std::cerr << fp_s1.count() << "s" << std::endl;
  
  QMatrix a2(500, 12);
  QMatrix b2(30000, 500);  
  Matrix c2;
            
  std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
  for(int i = 0; i < 100; ++i)
    QProd(context, c2, b2, a2);
  std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> fp_s2 = end2 - start2;
  std::cerr << fp_s2.count() << "s" << std::endl;

  Matrix af(12, 500);
  Matrix bf(500, 30000);  
  Matrix cf;
            
  std::chrono::steady_clock::time_point startf = std::chrono::steady_clock::now();
  for(int i = 0; i < 100; ++i)
    Prod(cf, af, bf);
  std::chrono::steady_clock::time_point endf = std::chrono::steady_clock::now();
  std::chrono::duration<double> fp_sf = endf - startf;
  std::cerr << fp_sf.count() << "s" << std::endl;
  
  Matrix af2(500, 12);
  Matrix bf2(30000, 500);  
  Matrix cf2;
            
  std::chrono::steady_clock::time_point startf2 = std::chrono::steady_clock::now();
  for(int i = 0; i < 100; ++i)
    Prod(cf2, bf2, af2);
  std::chrono::steady_clock::time_point endf2 = std::chrono::steady_clock::now();
  std::chrono::duration<double> fp_sf2 = endf2 - startf2;
  std::cerr << fp_sf2.count() << "s" << std::endl;
  
  //auto scorers = God::GetScorers(0);
  //EncoderDecoder& encdec = *std::static_pointer_cast<EncoderDecoder>(scorers[0]);
  //Encoder& encoder = encdec.GetEncoder();
  //Decoder& decoder = encdec.GetDecoder();
  //
  //Sentence s(0, "das ist ein kleiner Test .");
  //for(auto& w : s.GetWords())
  //  std::cerr << w << std::endl;
  //
  //std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  //for(size_t i = 0; i < 1; ++i) {
  //  mblas::Matrix context;
  //  encoder.GetContext(s.GetWords(), context);
  //  mblas::Debug(context);
  //}
  //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  //std::chrono::duration<double> fp_s = end - start;
  //LOG(progress) << fp_s.count() << "s";

  return 0;
}
