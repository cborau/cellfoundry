#ifndef CELLFOUNDRY_VARIANT_IDS_CUH
#define CELLFOUNDRY_VARIANT_IDS_CUH

// Claim one custom id from a managed variant population. `last` is its Int
// macro counter; end is the EXCLUSIVE range end declared by add_population().
// CAS performs an atomic read as well as allocation, avoiding a non-atomic
// macro read mixed with writes in the same layer. Each successful CAS is the
// atomic allocation point; the whole retry loop is not a single GPU instruction.
// Exhaustion sets a sticky flag for the mandatory host capacity check, returns
// -1, and leaves the counter bounded. A failed claim must not create an agent.
template<typename Counter, typename Flag>
__device__ __forceinline__ int cellfoundry_claim_variant_id(
        Counter &last, const int end, Flag &exhausted) {
    int observed = last.CAS(0, 0);
    while (observed < end - 1) {
        const int previous = last.CAS(observed, observed + 1);
        if (previous == observed) {
            return observed + 1;
        }
        observed = previous;
    }
    exhausted.exchange(1);
    return -1;
}

#endif
