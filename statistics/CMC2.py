import numpy as np

def calculate_cmc(gait_cycles):
    """
    gait_cycles: List of numpy arrays where each array contains the measured waveform data for a gait cycle.
                 Each array has a shape of (P, F), where
                 P = number of protocols and F = number of frames for that gait cycle (frames may vary).
    
    Returns:
        - The computed CMC value (if 1 - (normalized numerator/denominator) is negative, a complex value may be returned).
    
    [Calculation Process]
    1. For each gait cycle, using the waveform data from each protocol, compute:
       (a) the per-frame mean waveform (Ȳ₍g,f₎), and
       (b) the grand mean (Ȳg), which is the average of the per-frame means.
    2. For each gait cycle:
       (a) Numerator: the sum of squared differences between each protocol's waveform and the per-frame mean waveform,
           → num = ΣₚΣ_f (Y₍g,p,f₎ − Ȳ₍g,f₎)²
       (b) Denominator: the sum of squared differences between each protocol's waveform and the grand mean,
           → den = ΣₚΣ_f (Y₍g,p,f₎ − Ȳg)²
    3. Normalize each gait cycle by its degrees of freedom:
           num_norm = num / (F * (P − 1))
           den_norm = den / (F * P)
    4. Sum the normalized values over all gait cycles, then compute:
           ratio = (Σ num_norm) / (Σ den_norm)
    5. Final CMC = √(1 − ratio)
       (Note: if (1 − ratio) is negative, the result may be complex.)
    """
    num_total = 0.0  # Sum of normalized numerators for all gait cycles
    den_total = 0.0  # Sum of normalized denominators for all gait cycles

    # Process each gait cycle
    for cycle in gait_cycles:
        # Each cycle's shape: (P, F)
        # P: number of protocols, F: number of frames
        P, F = cycle.shape

        # [Step 1] Compute the per-frame mean waveform (Ȳ₍g,f₎)
        # Calculate the mean across protocols for each frame → shape: (F,)
        Y_bar_f = np.mean(cycle, axis=0)

        # [Step 2] Compute the grand mean (Ȳg): the average of the per-frame means
        Y_bar = np.mean(Y_bar_f)

        # [Step 3] Compute the numerator: sum of squared differences from the per-frame mean
        num = np.sum((cycle - Y_bar_f) ** 2)

        # [Step 4] Compute the denominator: sum of squared differences from the grand mean
        den = np.sum((cycle - Y_bar) ** 2)

        # [Step 5] Normalize by degrees of freedom: divide by F and (P - 1) for the numerator, and by F and P for the denominator
        num_norm = num / (F * (P - 1))
        den_norm = den / (F * P)

        # Accumulate the normalized values
        num_total += num_norm
        den_total += den_norm

    # [Step 6] Calculate the overall ratio across all gait cycles
    ratio = num_total / den_total

    # [Step 7] Compute CMC: square root of (1 - ratio)
    # If (1 - ratio) is negative, the result will be complex, so we use np.complex128.
    cmc_value = np.sqrt(np.complex128(1 - ratio))
    
    return cmc_value