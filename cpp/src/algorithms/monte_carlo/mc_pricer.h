#ifndef QK_MC_PRICER_H
#define QK_MC_PRICER_H

#include <quantkernel/qk_abi.h>

namespace qk {

/** 
 * Batch Monte Carlo pricer for European vanilla options.
 * [quantstart.com] (https://quantstart.com/articles/European-vanilla-option-pricing-with-C-via-Monte-Carlo-methods/)
*/
void mc_price_batch(const QKMCInput& input, QKMCOutput& output);

} // namespace qk

#endif /* QK_MC_PRICER_H */
