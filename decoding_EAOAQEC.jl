using LinearAlgebra

# code to generate the symplectic vectors
"""
    get_symplectic_vector(v1::Vector{Int}, v2::Vector{Int}, n::Int)

Returns a BitVector of length 2n where:
- Indices from `v1` (X-locations) are set to 1.
- Indices from `v2` shifted by `n` (Z-locations) are set to 1.
"""
function get_symplectic_vector(v1::Vector{Int}, v2::Vector{Int}, n::Int)
    # Initialize a bit vector of length 2n with all zeros
    symplectic_vec = falses(2n)
    
    # Set bits for the X-components (or Y)
    # These correspond to indices 1 to n
    for idx in v1
        if 1 <= idx <= n
            symplectic_vec[idx] = true
        else
            throw(BoundsError("Index $idx in the first vector exceeds n=$n"))
        end
    end
    
    # Set bits for the Z-components (or Y)
    # These correspond to indices (v2 + n), ranging from n+1 to 2n
    for idx in v2
        if 1 <= idx <= n
            symplectic_vec[idx + n] = true
        else
            throw(BoundsError("Index $idx in the second vector exceeds n=$n"))
        end
    end
    
    return symplectic_vec
end

"""
    generate_X_error(indices::Vector{Int}, n::Int)

Returns a symplectic BitVector of length 2n for Pauli X errors 
occurring at the specified qubit indices.
"""
function generate_X_error(indices::Vector{Int}, n::Int)
    # X errors only have support in the first n bits (v1)
    return get_symplectic_vector(indices, Int[], n)
end

"""
    generate_Y_error(indices::Vector{Int}, n::Int)

Returns a symplectic BitVector of length 2n for Pauli Y errors 
occurring at the specified qubit indices.
"""
function generate_Y_error(indices::Vector{Int}, n::Int)
    # Y errors have support in both the X (v1) and Z (v2) components
    return get_symplectic_vector(indices, indices, n)
end

"""
    generate_Z_error(indices::Vector{Int}, n::Int)

Returns a symplectic BitVector of length 2n for Pauli Z errors 
occurring at the specified qubit indices.
"""
function generate_Z_error(indices::Vector{Int}, n::Int)
    # Z errors only have support in the last n bits (v2)
    return get_symplectic_vector(Int[], indices, n)
end

# Obtain the syndrome based on the error vector
"""
    get_syndrome(S::AbstractMatrix, error_vec::AbstractVector)

Computes the syndrome vector for a given stabilizer matrix S and error vector error_vec.
- S: m x 2n matrix (Symplectic representation of generators)
- error_vec: 2n length vector (Symplectic representation of error)
"""
function get_syndrome(S::AbstractMatrix, error_vec::AbstractVector)
    # n is half the number of columns
    m, dual_n = size(S)
    n = div(dual_n, 2)
    
    # Ensure error vector matches the stabilizer dimensions
    if length(error_vec) != dual_n
        throw(DimensionMismatch("Error vector length must match number of columns in S."))
    end

    # Extract X and Z parts of the stabilizers
    X_s = S[:, 1:n]
    Z_s = S[:, n+1:end]
    
    # Extract X and Z parts of the error
    x_e = error_vec[1:n]
    z_e = error_vec[n+1:end]
    
    # Compute syndrome: (X_s * z_e + Z_s * x_e) mod 2
    # In Julia, BitMatrix multiplication doesn't automatically mod 2, 
    # so we use XOR and bitwise logic.
    syndrome = (X_s * z_e .% 2) .⊻ (Z_s * x_e .% 2)
    
    return BitVector(syndrome)
end

# Compute depolarizing probability
"""
Calculate the probability of a specific Pauli error with Hamming weight 'w' 
occurring on 'n' qubits, given a depolarizing error rate 'p'.

Parameters:
- w: Hamming weight (number of non-identity Pauli operators in the string).
- n: Total number of physical qubits.
- p: The physical depolarizing error probability (0 <= p <= 1).
"""
pauli_prob(w, n, p) = (1 - p)^(n - w) * (p / 3)^w

#Compute the probability associated with the coset
"""
    calculate_coset_probability(S, G, l, t, p)

Computes the sum of probabilities for the coset C_t = {v + l + t | v ∈ V} 
using BitVectors for efficiency.
"""
function calculate_coset_probability(S::BitMatrix, G::BitMatrix, l::BitVector, t::BitVector, p::Float64)
    # 1. Filter rows of S
    # We check if all bits at mask_indices are 'false' (0)
    mask_indices = [11, 12, 13, 24, 25, 26]
    valid_rows = [!any(S[i, mask_indices]) for i in 1:size(S, 1)]
    S_filtered = S[valid_rows, :]

    # 2. Concatenate to form basis
    basis = vcat(G, S_filtered)
    num_generators, vector_length = size(basis)
    n_physical = div(vector_length, 2)

    # 3. Optimized Pauli weight for BitVectors
    # A Pauli error at qubit i occurs if either X_i or Z_i is 1.
    function pauli_weight(v::BitVector)
        # Slices in BitVectors are efficient, and .| (OR) is bit-parallel
        return sum(v[1:n_physical] .| v[n_physical+1:end])
    end

    # 4. Probability constants
    n_param = 10 
    # Pre-calculating these avoids repetitive power operations
    term_1_minus_p = 1 - p
    term_p_div_3 = p / 3
    
    pauli_prob(w) = (term_1_minus_p)^(n_param - w) * (term_p_div_3)^w

    total_sum_prob = 0.0
    
    # Pre-combine l and t to save one XOR operation inside the loop
    lt_combined = l .⊻ t

    # 5. Generate all 2^k vectors in the vector space V
    for i in 0:(2^num_generators - 1)
        # Pre-allocate or reset v for each iteration
        v = BitVector(zeros(vector_length))
        
        for j in 1:num_generators
            if (i >> (j - 1)) & 1 == 1
                # .⊻ is bitwise XOR (addition mod 2)
                v .⊻= basis[j, :]
            end
        end

        # Calculate element in coset: c = v ⊕ (l ⊕ t)
        c = v .⊻ lt_combined

        w = pauli_weight(c)
        total_sum_prob += pauli_prob(w)
    end

    return total_sum_prob
end

function lowest_weight_coset_vector(l::AbstractVector, t::AbstractVector, S::AbstractMatrix, G::AbstractMatrix)
    # 1. Filter rows of S where columns 11, 12, 13, 24, 25 and 26 are all 0
    valid_rows = Int[]
    for i in 1:size(S, 1)
        if S[i, 11] == 0 && S[i, 12] == 0 && S[i, 13] == 0 && S[i, 24] == 0 && S[i, 25] == 0 && S[i, 26] == 0
            push!(valid_rows, i)
        end
    end
    S_filtered = S[valid_rows, :]
    
    # 2. Combine filtered S and G
    M = vcat(S_filtered, G)
    
    # 3. Find a basis for the row space of M over GF(2) to remove redundant rows
    M_basis = get_gf2_basis(M)
    k = size(M_basis, 1)
    
    # Base vector: (l + t) mod 2
    base_vec = (l .+ t) .% 2
    
    if k == 0
        return base_vec
    end
    
    min_wt = length(base_vec) + 1
    best_vec = copy(base_vec)
    best_v = copy(base_vec)
    best_v .= 0
    
    # 4. Search the row space to find the minimum weight vector
    # Iterating through all 2^k possible linear combinations over GF(2)
    for i in 0:(2^k - 1)
        v = zeros(Int, size(M_basis, 2))
        
        # Generate the linear combination for the current iteration
        for j in 1:k
            if ((i >> (j - 1)) & 1) == 1
                v .= (v .+ M_basis[j, :]) .% 2
            end
        end
        
        # Add the generated row space vector to (l + t)
        candidate = (v .+ base_vec) .% 2
        wt = sum(candidate)
        
        # Update minimum
        if wt < min_wt
            min_wt = wt
            best_vec = candidate
            best_v = v
            
            # Short-circuit if we hit the absolute minimum possible weight
            if min_wt == 0
                break
            end
        end
    end
    println("base_vec = ", print_pauli_operators(base_vec', true))
    println("best_v = ", print_pauli_operators(best_v', true))
    println("min wt = ", min_wt)
    return best_vec
end

# Helper function to compute the row space basis over GF(2)
function get_gf2_basis(M::AbstractMatrix)
    A = copy(M) .% 2
    rows, cols = size(A)
    r = 1
    for c in 1:cols
        if r > rows
            break
        end
        
        # Find pivot
        pivot = findfirst(x -> x != 0, A[r:end, c])
        
        if pivot !== nothing
            pivot += (r - 1)
            # Swap rows if necessary
            if pivot != r
                A[r, :], A[pivot, :] = A[pivot, :], A[r, :]
            end
            
            # Eliminate other 1s in the current column
            for i in 1:rows
                if i != r && A[i, c] != 0
                    A[i, :] .= (A[i, :] .+ A[r, :]) .% 2
                end
            end
            r += 1
        end
    end
    
    # Return only the non-zero rows (the linearly independent basis)
    return A[1:(r-1), :]
end

# Instance parameters:

# Stabilizer Generators (8 x 26)
S = [
    0 0 0 0 0 0 0 0 0 0 0 0 0  1 0 0 1 1 0 0 1 1 0 1 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0  0 1 0 1 0 1 0 1 0 1 0 1 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0  1 0 1 0 1 0 0 0 0 0 0 0 1;
    1 0 0 1 1 0 0 1 1 0 1 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 1 0 1 0 1 0 1 0 1 0 1 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    1 0 1 0 1 0 0 0 0 0 0 0 1  0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 1 1 1 1 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 1 1 1 1 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0  0 0 0 0 0 0 1 1 1 1 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0  0 0 1 1 1 1 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0  1 1 1 1 0 0 0 0 0 0 0 0 0
]

# Logical Operators (2 x 26)
L = [
    0 1 1 0 1 0 1 1 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0  0 1 1 0 1 0 1 1 0 0 0 0 0
]

# Gauge Generators (6 x 26)
G = [
    1 1 1 1 0 0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0  1 0 1 1 0 0 0 1 1 0 0 0 0
]

# Coset transversal matrix T
T_0 = [
    0 0 0 0 0 0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 1 0 0 0 0 1 0 0 1 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    1 0 0 0 0 0 1 0 1 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0  1 0 0 0 0 0 1 0 1 0 0 0 0
]

# Depolarizing probability p
p = 0.005

# coset_representative dictionary t_s_dict

using Combinatorics
using LinearAlgebra

"""
    generate_min_weight_syndrome_table(S::AbstractMatrix, max_weight::Int=4)

Returns (status, missing_syndromes, lookup_table).
- status: 0 if all 2^m syndromes are found, -1 otherwise.
- missing_syndromes: List of BitVectors not covered by errors up to max_weight.
- lookup_table: Dict{BitVector, BitVector} mapping syndrome -> lowest weight error.
"""
function generate_min_weight_syndrome_table(S::AbstractMatrix, ebits::Int, max_weight::Int=4)
    m, dual_n = size(S)
    n = div(dual_n, 2)
    
    # Pre-split S for faster syndrome calculation via symplectic inner product
    X_s = S[:, 1:n]
    Z_s = S[:, n+1:end]
    
    lookup_table = Dict{BitVector, BitVector}()
    
    # 1. Iterate through weights 0 to max_weight
    # Weight 0 is the identity (no error, zero syndrome)
    for w in 0:max_weight
        # Update: only consider indices from 1 up to (n - ebits)
        # This assumes the ebits are the LAST indices in the n-qubit block.
        for indices in combinations(1:(n - ebits), w)
            # Generate all 3^w Pauli combinations for the chosen indices
            # 1=X, 2=Y, 3=Z
            for pauli_types in Iterators.product(fill(1:3, w)...)
                error_vec = falses(2n) # renamed from 'e' to avoid conflict
                for (i, p_type) in enumerate(pauli_types)
                    idx = indices[i]
                    if p_type == 1      # X
                        error_vec[idx] = true
                    elseif p_type == 2  # Y
                        error_vec[idx] = true
                        error_vec[idx + n] = true
                    elseif p_type == 3  # Z
                        error_vec[idx + n] = true
                    end
                end
                
                # Compute syndrome: s = (X_s * z_e + Z_s * x_e) mod 2
                x_e = error_vec[1:n]
                z_e = error_vec[n+1:end]
                syndrome = BitVector((X_s * z_e .% 2) .⊻ (Z_s * x_e .% 2))
                
                # Store if this is the first (and thus lowest weight) time seeing this syndrome
                if !haskey(lookup_table, syndrome)
                    lookup_table[syndrome] = error_vec
                end
            end
        end
    end
    
    # 2. Check if all possible syndromes were found
    total_possible = 2^m
    if length(lookup_table) == total_possible
        return 0, BitVector[], lookup_table
    else
        # Identify missing syndromes
        missing_syndromes = BitVector[]
        for i in 0:(total_possible - 1)
            # Convert integer to bit vector of length m
            s_vec = BitVector(digits(i, base=2, pad=m))
            if !haskey(lookup_table, s_vec)
                push!(missing_syndromes, s_vec)
            end
        end
        return -1, missing_syndromes, lookup_table
    end
end


function print_pauli_operators(M::Union{AbstractMatrix, AbstractVector}, return_vector::Bool = false)
    M = (M isa AbstractVector) ? reshape(M, 1, :) : M
    rows, cols = size(M)
    
    # Ensure the matrix has an even number of columns (2n)
    if !iseven(cols)
        throw(ArgumentError("The matrix must have an even number of columns (2n). There are $(cols) columns."))
    end
    
    n = cols ÷ 2
    
    for i in 1:rows
        pauli_chars = Char[]
        
        for j in 1:n
            x = M[i, j]
            z = M[i, j + n]
            
            if x == 0 && z == 0
                push!(pauli_chars, 'I')
            elseif x == 1 && z == 0
                push!(pauli_chars, 'X')
            elseif x == 0 && z == 1
                push!(pauli_chars, 'Z')
            elseif x == 1 && z == 1
                push!(pauli_chars, 'Y')
            else
                push!(pauli_chars, '?') # Fallback for non-binary matrix entries
            end
        end
        
        # Join the characters into a string and print
        pauli_str = join(pauli_chars)
        if rows == 1 && return_vector
            return pauli_str
        end
        println("Row $i: $pauli_str")
    end
end

using LinearAlgebra

"""
    binary_row_space(A::Matrix{Int})

Generates all unique vectors in the row space of a binary matrix A over GF(2).
"""
function binary_row_space(A::Matrix{Int})
    # 1. Find the Basis using Gaussian Elimination over GF(2)
    rows, cols = size(A)
    basis = []
    temp_A = copy(A) .% 2 # Ensure binary
    
    pivot_row = 1
    for j in 1:cols
        pivot_row > rows && break
        
        # Find a row with a 1 in the current column
        idx = findfirst(x -> x == 1, temp_A[pivot_row:end, j])
        if idx !== nothing
            idx += pivot_row - 1
            # Swap current row with the found row
            temp_A[pivot_row, :], temp_A[idx, :] = temp_A[idx, :], temp_A[pivot_row, :]
            
            # Eliminate other 1s in this column using XOR
            for i in 1:rows
                if i != pivot_row && temp_A[i, j] == 1
                    temp_A[i, :] .= (temp_A[i, :] .+ temp_A[pivot_row, :]) .% 2
                end
            end
            push!(basis, temp_A[pivot_row, :])
            pivot_row += 1
        end
    end

    # 2. Generate all linear combinations of the basis vectors
    # If basis has n vectors, there are 2^n unique vectors in the row space
    n = length(basis)
    row_space = Vector{Vector{Int}}()
    
    for i in 0:(2^n - 1)
        # Create a combination based on the binary representation of i
        combo = zeros(Int, cols)
        bits = digits(i, base=2, pad=n)
        for (idx, bit) in enumerate(bits)
            if bit == 1
                combo = (combo .+ basis[idx]) .% 2
            end
        end
        push!(row_space, combo)
    end
    
    return row_space
end

function rank_gf2(M::AbstractMatrix)
    A = copy(M) .% 2
    rows, cols = size(A)
    r = 1
    for c in 1:cols
        if r > rows break end
        pivot_idx = findfirst(x -> x != 0, A[r:end, c])
        if pivot_idx !== nothing
            pivot_idx += (r - 1)
            if pivot_idx != r
                A[r, :], A[pivot_idx, :] = A[pivot_idx, :], A[r, :]
            end
            for i in 1:rows
                if i != r && A[i, c] != 0
                    A[i, :] .= (A[i, :] .+ A[r, :]) .% 2
                end
            end
            r += 1
        end
    end
    return r - 1
end


# Example usage:
# A = [1 0 1; 1 1 0; 0 1 1]
# space = binary_row_space(A)

# Decode the code

"""
    decode_EAOAQEC_single_level(S, G, L, T_0, s_bar, p, t_s_dict)

Performs decoding for an Entanglement-Assisted Operator Quantum Error Correction (EAOAQEC) code.
Returns the recovery operator, the optimal logical operator l*, and the transversal element t*.
"""
function decode_EAOAQEC_single_level(s_bar, S, G, L, T_0, p, t_s_dict)
    # 1. Retrieve t_s_bar from the dictionary using the syndrome s_bar
    t_s_bar = BitVector(t_s_dict[s_bar])
    
    # Initialize trackers for the maximum probability search
    # We use 0 so that the first calculated probability will always be larger
    max_prob = 0
    l_star = BitVector(undef, 26)
    t_star = BitVector(undef, 26)
    
    num_T0_rows = size(T_0, 1)
    k_logical = size(L, 1)
    
    # 2. Iterate over all elements t in T_0 (rows of the matrix T_0)
    for i in 1:num_T0_rows
        t = BitVector(T_0[i, :])
        
        # 3. Iterate over the row space of L
        # There are 2^k elements in the row space of a k-row generator matrix
        for j in 0:(2^k_logical - 1)
            # Generate coefficients for the linear combination of rows
            coeffs = digits(j, base=2, pad=k_logical)
            
            # Compute element l in the row space: l = coeffs * L (mod 2)
            # We use transpose multiplication and ensure results are 0 or 1
            l = BitVector(vec(coeffs' * L) .% 2)
            
            # Prepare the second argument: (t + t_s_bar) mod 2
            # In Julia, .!= or .⊻ serves as a bitwise XOR for BitVectors
            t_combined = t .⊻ t_s_bar
            
            # 4. Calculate probability using the external function
            prob = calculate_coset_probability(BitMatrix(Bool.(S)), BitMatrix(Bool.(G)), l, t_combined, p)

            
            # 5. Check for the maximum probability
            println("prob = ", prob, ", error = ", print_pauli_operators((l .⊻ t_combined)', true))
            if prob > max_prob
                max_prob = prob
                l_star = copy(l)
                t_star = copy(t)
            end
        end
    end
    
    # 6. Compute the recovery operator: l* + t* + t_s_bar (mod 2)
    recovery_operator = l_star .⊻ t_star .⊻ t_s_bar
    println("max prob = ", max_prob, ", recovery operator = ", print_pauli_operators(recovery_operator, true), "\n\n")
    
    return recovery_operator, l_star, t_star, t_s_bar
end

function obtain_stabs_in_S_Q(S, T)
    commutativity_matrix = transpose(stack(get_syndrome(T, s) for s in eachrow(S)))
    zero_row_indices = findall(r -> all(iszero, r), eachrow(commutativity_matrix ))
    return zero_row_indices, (size(S, 1) - length(zero_row_indices)) == rank_gf2(commutativity_matrix[setdiff(1:size(S,1), zero_row_indices),:]) 
end

# Decode the code

"""
    decode_EAOAQEC_two_level(S, G, L, T_0, s_bar, p, t_s_dict)

Performs decoding for an Entanglement-Assisted Operator Quantum Error Correction (EAOAQEC) code.
Returns the recovery operator, the optimal logical operator l*, and the transversal element t*.
"""
function decode_EAOAQEC_two_level(s_bar, S, G, L, T_0, p, t_s_dict)

    # 1. Obtain the quantum stabilizers
    S_Q_indices, full_S_Q_returned = obtain_stabs_in_S_Q(S, T_0)
    if !full_S_Q_returned
        throw("The entire S_Q is not returned. The code for obtaining the entire S_Q is not implemented yet.")
    end
    
    # Retrieve t_s_q_bar from the dictionary using the syndrome s_q_bar
    s_q_bar = [i in S_Q_indices ? s_bar[i] : 0 for i in eachindex(s_bar)]
    t_s_q_bar = BitVector(t_s_dict[s_q_bar]) 
    
    # STAGE 1
    
    # Initialize trackers for the maximum probability search
    # We use 0 so that the first calculated probability will always be larger
    max_prob = 0
    l_star = BitVector(undef, 26)
    t_star = BitVector(undef, 26)
    
    T_0_span = binary_row_space(T_0)
    num_T0_span_rows = size(T_0_span, 1)
    k_logical = size(L, 1)
    #println("T_0_span = ", T_0_span)
    
    # 2. Iterate over all elements t in T_0_span (rowspace of the matrix T_0)
    for i in 1:num_T0_span_rows
        t = BitVector(T_0_span[i])
        
        # 3. Iterate over the row space of L
        # There are 2^k elements in the row space of a k-row generator matrix
        for j in 0:(2^k_logical - 1)
            # Generate coefficients for the linear combination of rows
            coeffs = digits(j, base=2, pad=k_logical)
            
            # Compute element l in the row space: l = coeffs * L (mod 2)
            # We use transpose multiplication and ensure results are 0 or 1
            l = BitVector(vec(coeffs' * L) .% 2)
            
            # Prepare the second argument: (t + t_s_bar) mod 2
            # In Julia, .!= or .⊻ serves as a bitwise XOR for BitVectors
            t_combined = t .⊻ t_s_q_bar
            
            # 4. Calculate probability using the external function
            prob = calculate_coset_probability(BitMatrix(Bool.(S)), BitMatrix(Bool.(G)), l, t_combined, p)
            
            # 5. Check for the maximum probability
            println("prob = ", prob, ", error = ", print_pauli_operators((l .⊻ t_combined)', true))
            if prob > max_prob
                max_prob = prob
                l_star = copy(l)
                t_star = copy(t)
            end
        end
    end
    
    s_C_q_bar = [i in S_Q_indices ? 0 : s_bar[i] for i in eachindex(s_bar)]
    t_s_C_q_bar = BitVector(t_s_dict[s_C_q_bar])
    
    println("Stage 1 completed.\n", "Stage 1 Recovery operator: ", print_pauli_operators((l_star .⊻ t_star .⊻ t_s_q_bar)', true), ", prob = ", max_prob, "\n")
    println("Syndrome for recovery after stage 1 = ", findall(!iszero, get_syndrome(S, l_star .⊻ t_star .⊻ t_s_q_bar)))
    println("Syndrome for recovery corresponds to coset transversals? ", any(get_syndrome(S, row) == (get_syndrome(S, l_star .⊻ t_star .⊻ t_s_q_bar) .⊻ s_bar) for row in eachrow(T_0)))
    println("Desirable syndrome support: ")
    for row in eachrow(T_0)
        println(findall(!iszero, get_syndrome(S, l_star .⊻ t_star .⊻ t_s_q_bar .⊻ row)))
    end
    if any(get_syndrome(S, row) == (get_syndrome(S, l_star .⊻ t_star .⊻ t_s_q_bar) .⊻ s_bar) for row in eachrow(T_0))
        return l_star .⊻ t_star .⊻ t_s_q_bar, l_star, t_star .⊻ t_s_q_bar, t_star, t_s_q_bar, nothing, nothing
    end
    
    # STAGE 2
    
    # Initialize trackers for the maximum probability search
    # We use 0 so that the first calculated probability will always be larger
    max_prob = 0
    t_q_star_star = BitVector(undef, 26)
    
    T_0_q = T_0[2:end,:]
    num_T0_q_rows = size(T_0_q, 1)
    k_logical = size(L, 1)
    
    # 2. Iterate over all elements t in T_0_q (rowspace of the matrix T_0)
    for i in 1:num_T0_q_rows
        t = BitVector(T_0_q[i,:])        
            # Prepare the second argument: (t + t_s_bar) mod 2
            # In Julia, .!= or .⊻ serves as a bitwise XOR for BitVectors
            t_combined = t .⊻ t_s_C_q_bar .⊻ t_s_q_bar
            println("Syndrome for t_combined = ", findall(!iszero, get_syndrome(S, t_combined)))
            
            # 4. Calculate probability using the external function
            prob = calculate_coset_probability(BitMatrix(Bool.(S)), BitMatrix(Bool.(G)), l_star, t_combined, p)
 
            # 5. Check for the maximum probability
            println("prob = ", prob, ", error = ", print_pauli_operators((l_star .⊻ t_combined)', true))
            if prob > max_prob
                max_prob = prob
                t_q_star_star = copy(t)
            end
    end
    # 6. Compute the recovery operator: l* + t* + t_s_bar (mod 2)
    recovery_operator = l_star .⊻ t_q_star_star .⊻ t_s_C_q_bar .⊻ t_s_q_bar
     println("recovery_operator = ", print_pauli_operators(recovery_operator,true))
println("l_star = ", print_pauli_operators(l_star', true))
println("t_star = ", print_pauli_operators(t_star', true))
println("t_q_star_star = ", print_pauli_operators(t_q_star_star', true))
println("t_s_C_q_bar = ", print_pauli_operators(t_s_C_q_bar', true))
println("Syndrome for t_s_C_q_bar = ", findall(!iszero,get_syndrome(S, t_s_C_q_bar)))
println("t_s_q_bar = ", print_pauli_operators(t_s_q_bar', true))
    println("max prob = ", max_prob, ", recovery operator = ", print_pauli_operators(recovery_operator', true), "\n\n")
    
    return recovery_operator, l_star, t_q_star_star .⊻ t_s_C_q_bar .⊻ t_s_q_bar, t_star, t_s_q_bar, t_q_star_star, t_s_C_q_bar
end

# Decode the code

"""
    decode_EAOAQEC_two_level(S, G, L, T_0, s_bar, p, t_s_dict)

Performs decoding for an Entanglement-Assisted Operator Quantum Error Correction (EAOAQEC) code.
Returns the recovery operator, the optimal logical operator l*, and the transversal element t*.
"""
function decode_EAOAQEC_two_level_type2(s_bar, S, G, L, T_0, p, t_s_dict)

    # 1. Obtain the quantum stabilizers
    S_Q_indices, full_S_Q_returned = obtain_stabs_in_S_Q(S, T_0)
    if !full_S_Q_returned
        throw("The entire S_Q is not returned. The code for obtaining the entire S_Q is not implemented yet.")
    end
    
    # Retrieve t_s_q_bar and t_s_C_q_bar from the dictionary using the syndrome s_q_bar and s_C_q_bar
    s_q_bar = [i in S_Q_indices ? s_bar[i] : 0 for i in eachindex(s_bar)]
    t_s_q_bar = BitVector(t_s_dict[s_q_bar]) 

    s_C_q_bar = [i in S_Q_indices ? 0 : s_bar[i] for i in eachindex(s_bar)]
    t_s_C_q_bar = BitVector(t_s_dict[s_C_q_bar])
    
    # STAGE 1
    
    # Initialize trackers for the maximum probability search
    # We use 0 so that the first calculated probability will always be larger
    max_prob = 0
    t_star_star_star = BitVector(undef, 26)
    k_logical = size(L, 1)
    num_T0_rows = size(T_0, 1)
    
    for i in 1:num_T0_rows
        t = BitVector(T_0[i,:])        
            # Prepare the second argument: (t + t_s_bar) mod 2
            # In Julia, .!= or .⊻ serves as a bitwise XOR for BitVectors
            t_combined = t .⊻ t_s_C_q_bar .⊻ t_s_q_bar
            println("Syndrome for t_combined = ", findall(!iszero, get_syndrome(S, t_combined)))
            
            prob = 0
            # Calculate probability using the external function
            for j in 0:(2^k_logical - 1)
                # Generate coefficients for the linear combination of rows
                coeffs = digits(j, base=2, pad=k_logical)
            
                # Compute element l in the row space: l = coeffs * L (mod 2)
                # We use transpose multiplication and ensure results are 0 or 1
                l = BitVector(vec(coeffs' * L) .% 2)
                prob += calculate_coset_probability(BitMatrix(Bool.(S)), BitMatrix(Bool.(G)), l, t_combined, p)
            end
            # 5. Check for the maximum probability
            println("prob = ", prob, ", error = ", print_pauli_operators((l.⊻ t_combined)', true))
            if prob > max_prob
                max_prob = prob
                t_star_star_star = copy(t)
            end
    end
    
    l_star_star_star = BitVector(undef, 26)
        
    # STAGE 2
    
    # Initialize trackers for the maximum probability search
    # We use 0 so that the first calculated probability will always be larger
    l_star_star_star = BitVector(undef, 26)
    max_prob = 0
    t_combined = t_star_star_star .⊻ t_s_C_q_bar .⊻ t_s_q_bar
        # 2. Iterate over all elements t in T_0_q (rowspace of the matrix T_0)
            for j in 0:(2^k_logical - 1)
                # Generate coefficients for the linear combination of rows
                coeffs = digits(j, base=2, pad=k_logical)
             
                # Compute element l in the row space: l = coeffs * L (mod 2)
                # We use transpose multiplication and ensure results are 0 or 1
                l = BitVector(vec(coeffs' * L) .% 2)
                prob = calculate_coset_probability(BitMatrix(Bool.(S)), BitMatrix(Bool.(G)), l, t_combined, p)
            end
            # 5. Check for the maximum probability
            println("prob = ", prob, ", error = ", print_pauli_operators((l .⊻ t_combined)', true))
            if prob > max_prob
                max_prob = prob
                t_star_star_star = copy(t)
            end
    # 6. Compute the recovery operator: l* + t* + t_s_bar (mod 2)
    recovery_operator = l_star_star_star .⊻ t_star_star_star .⊻ t_s_C_q_bar .⊻ t_s_q_bar
println("l_star_star_star = ", print_pauli_operators(l_star_star_star', true))
println("t_star_star_star = ", print_pauli_operators(t_star_star_star', true))
println("t_s_C_q_bar = ", print_pauli_operators(t_s_C_q_bar', true))
println("Syndrome for t_s_C_q_bar = ", findall(!iszero,get_syndrome(S, t_s_C_q_bar)))
println("t_s_q_bar = ", print_pauli_operators(t_s_q_bar', true))
println("max prob = ", max_prob, ", recovery operator = ", print_pauli_operators(recovery_operator', true), "\n\n")
    
    return recovery_operator, l_star_star_star, t_star_star_star .⊻ t_s_C_q_bar .⊻ t_s_q_bar, t_star_star_star, t_s_q_bar, t_s_C_q_bar
end

function decode_EAOAQEC_two_level_type2(s_bar, S, G, L, T_0, p, t_s_dict)

    # 1. Obtain the quantum stabilizers and verify the entire S_Q is returned
    S_Q_indices, full_S_Q_returned = obtain_stabs_in_S_Q(S, T_0)
    if !full_S_Q_returned
        throw("The entire S_Q is not returned. The code for obtaining the entire S_Q is not implemented yet.")
    end
    
    # 2. Extract quantum syndrome s_q_bar and lookup corresponding pure error t_s_q_bar
    s_q_bar = [i in S_Q_indices ? s_bar[i] : 0 for i in eachindex(s_bar)]
    t_s_q_bar = BitVector(t_s_dict[s_q_bar]) 

    # 3. Extract classical syndrome s_C_q_bar and lookup corresponding pure error t_s_C_q_bar
    s_C_q_bar = [i in S_Q_indices ? 0 : s_bar[i] for i in eachindex(s_bar)]
    t_s_C_q_bar = BitVector(t_s_dict[s_C_q_bar])
    
    # ==========================================
    # STAGE 1: Find the optimal transversal operator
    # ==========================================
    
    # Initialize trackers to find the optimal transversal operator (t_star_star_star)
    max_prob = 0
    t_star_star_star = BitVector(undef, 26)
    k_logical = size(L, 1)
    num_T0_rows = size(T_0, 1)
    
    # Iterate through each row in T_0
    for i in 1:num_T0_rows
        t = BitVector(T_0[i,:])        
        
        # Calculate combined pure error: (t + t_s_C_q_bar + t_s_q_bar) mod 2
        t_combined = t .⊻ t_s_C_q_bar .⊻ t_s_q_bar
        println("Syndrome for t_combined = ", findall(!iszero, get_syndrome(S, t_combined)))
        
        prob = 0
        local l # Declare l as local so it can be accessed by the print statement later
        
        # Accumulate total probability over all possible logical operators
        for j in 0:(2^k_logical - 1)
            # Generate binary coefficients for the linear combination of rows
            coeffs = digits(j, base=2, pad=k_logical)
        
            # Compute element l in the row space: l = coeffs^T * L (mod 2)
            l = BitVector(vec(coeffs' * L) .% 2)
            prob += calculate_coset_probability(BitMatrix(Bool.(S)), BitMatrix(Bool.(G)), l, t_combined, p)
        end
        
        # Print total accumulated probability for this gauge operator t
        println("prob = ", prob, ", error = ", print_pauli_operators((l .⊻ t_combined)', true))
        
        # Update the maximum probability and store the optimal t
        if prob > max_prob
            max_prob = prob
            t_star_star_star = copy(t)
        end
    end
        
    # ==========================================
    # STAGE 2: Find the optimal logical operator
    # ==========================================
    
    # Reset trackers to find the optimal logical operator (l_star_star_star)
    l_star_star_star = BitVector(undef, 26)
    max_prob = 0
    
    # Recalculate combined pure error using the optimal t found in Stage 1
    t_combined = t_star_star_star .⊻ t_s_C_q_bar .⊻ t_s_q_bar
    
    # Iterate over all logical operators to find the one with maximum probability
    for j in 0:(2^k_logical - 1)
        # Generate binary coefficients for the linear combination of rows
        coeffs = digits(j, base=2, pad=k_logical)
     
        # Compute element l in the row space: l = coeffs^T * L (mod 2)
        l = BitVector(vec(coeffs' * L) .% 2)
        
        # Calculate the specific coset probability for this logical operator l
        prob = calculate_coset_probability(BitMatrix(Bool.(S)), BitMatrix(Bool.(G)), l, t_combined, p)
        
        println("prob = ", prob, ", error = ", print_pauli_operators((l .⊻ t_combined)', true))
        
        # Check if current probability is the maximum; if so, update max_prob and store optimal l
        if prob > max_prob
            max_prob = prob
            l_star_star_star = copy(l)
        end
    end
    
    # Compute the final recovery operator: l* + t* + t_s_C_q_bar + t_s_q_bar (mod 2)
    recovery_operator = l_star_star_star .⊻ t_star_star_star .⊻ t_s_C_q_bar .⊻ t_s_q_bar

    # Print final tracked elements
    println("l_star_star_star = ", print_pauli_operators(l_star_star_star', true))
    println("t_star_star_star = ", print_pauli_operators(t_star_star_star', true))
    println("t_s_C_q_bar = ", print_pauli_operators(t_s_C_q_bar', true))
    println("Syndrome for t_s_C_q_bar = ", findall(!iszero, get_syndrome(S, t_s_C_q_bar)))
    println("t_s_q_bar = ", print_pauli_operators(t_s_q_bar', true))
    println("max prob = ", max_prob, ", recovery operator = ", print_pauli_operators(recovery_operator', true), "\n\n")
    
    # Return the recovery operator alongside the extracted components
    return recovery_operator, l_star_star_star, t_combined, t_star_star_star, t_s_q_bar, t_s_C_q_bar
end


function main()
for m in 4:4
   # Generate the syndrome to coset transversal dictionary
    success, missing_syndrome, t_s_dict = generate_min_weight_syndrome_table(S,3,5)

    if success != -1
        println("\n\n\n m = ", m, "\n\n\n")
        # choose the errors
        #e1 = (generate_X_error([1, m], 13) + L[1,:]).%2
        e1 =  generate_X_error([2], 13)
        e2 = (generate_X_error([1, m], 13)).%2 

        # Obtain the syndromes
        s1 = get_syndrome(S, e1)
        s2 = get_syndrome(S, e2)

        println("\n\n Example 1: \n\n")
        println("e1 = ", print_pauli_operators(e1, true))
        println("s1 = ", findall(!iszero, s1))
 
        # Run the decoding solvers
        println("\n One level procedure: \n")
        r1, l1, t1, t_s_bar1 = decode_EAOAQEC_single_level(s1, S, G, L, T_0, p, t_s_dict)

        println("l1 = ", print_pauli_operators(l1, true))
        println("t1 = ", print_pauli_operators(t1, true))
        println("t_s_bar1 = ", print_pauli_operators(t_s_bar1, true))
        println("t1 + t_s_bar1 = ", print_pauli_operators((t1 + t_s_bar1).%2, true))
        println("r1 = ", print_pauli_operators(r1, true), ", lowest weight operator in coset: ", print_pauli_operators(lowest_weight_coset_vector(l1, t1 .⊻ t_s_bar1, S, G), true))

        
		# Check if the syndrome of the recovery operator is equal to that of the error
		@assert any(get_syndrome(S, row) == (get_syndrome(S, r1) .⊻ s1) for row in eachrow(T_0))

        println("\n","Two level decoding procedure:\n")
		r1_twolevel, l1_twolevel, t1_twolevel, t_star1, t_s_bar1_twolevel, t_q1, t_s_C_q_bar1 = decode_EAOAQEC_two_level(s1, S, G, L, T_0, p, t_s_dict)

        println("r1_twolevel = ", print_pauli_operators(r1_twolevel, true))
        println("l1_twolevel = ", print_pauli_operators(l1, true))   
        println("t1_twolevel = ", print_pauli_operators(t1, true))
        println("t_star1 = ", print_pauli_operators(t_star1, true))
        println("t_s_bar1_twolevel = ", print_pauli_operators(t_s_bar1_twolevel, true))
        isnothing(t_q1) || println("t_q1 = ", print_pauli_operators(t_q1, true))
        isnothing(t_s_C_q_bar1) || println("t_s_C_q_bar1 = ", print_pauli_operators(t_s_C_q_bar1, true))
        println("syndrome for r1_twolevel = ", findall(!iszero,get_syndrome(S, r1_twolevel)))
        println("r1_twolevel = ", print_pauli_operators(r1_twolevel, true), ", lowest weight operator in coset: ", print_pauli_operators(lowest_weight_coset_vector(l1, t1_twolevel, S, G), true))

        # Check if the syndrome of the recovery operator is equal to that of the error
        @assert any(get_syndrome(S, row) == (get_syndrome(S, r1_twolevel) .⊻ s1) for row in eachrow(T_0))

        println("\n", "Two level decoding procedure (Type 2):\n")
        r1_two_level_2, l1_twolevel_2, t1_twolevel_2, t_star_star_star, t_s_q_bar1_twolevel_2, t_s_C_q_bar1_twolevel_2 = decode_EAOAQEC_two_level_type2(s1, S, G, L, T_0, p, t_s_dict)

        println("r1_two_level_2 = ", print_pauli_operators(r1_two_level_2, true))
        println("l1_twolevel_2 = ", print_pauli_operators(l1_twolevel_2, true))   
        println("t1_twolevel_2 = ", print_pauli_operators(t1_twolevel_2, true))
        println("t_star_star_star = ", print_pauli_operators(t_star_star_star, true))
        
        isnothing(t_s_q_bar1_twolevel_2) || println("t_s_q_bar1_twolevel_2 = ", print_pauli_operators(t_s_q_bar1_twolevel_2, true))
        isnothing(t_s_C_q_bar1_twolevel_2) || println("t_s_C_q_bar1_twolevel_2 = ", print_pauli_operators(t_s_C_q_bar1_twolevel_2, true))
        
        println("syndrome for r1_two_level_2 = ", findall(!iszero, get_syndrome(S, r1_two_level_2)))
        println("r1_two_level_2 = ", print_pauli_operators(r1_two_level_2, true), ", lowest weight operator in coset: ", print_pauli_operators(lowest_weight_coset_vector(l1_twolevel_2, t1_twolevel_2, S, G), true))

        # Check if the syndrome of the recovery operator is equal to that of the error
        @assert any(get_syndrome(S, row) == (get_syndrome(S, r1_two_level_2) .⊻ s1) for row in eachrow(T_0))

        println("\n\n Example 2: \n\n")
        println("e2 = ", print_pauli_operators(e2, true))
        println("s2 = ", findall(!iszero, s2))

        # Run the decoding solvers
        println("One level decoding procedure:")
        r2, l2, t2, t_s_bar2 = decode_EAOAQEC_single_level(s2, S, G, L, T_0, p, t_s_dict)
        println("l2 = ", print_pauli_operators(l2, true))
        println("t2 = ", print_pauli_operators(t2, true))
        println("t_s_bar2 = ", print_pauli_operators(t_s_bar2, true))
        println("t2 + t_s_bar2 = ", print_pauli_operators((t2 + t_s_bar2).%2, true))
        println("r2 = ", print_pauli_operators(r2, true), ", lowest weight operator in coset: ", print_pauli_operators(lowest_weight_coset_vector(l2, t2 .⊻ t_s_bar2, S, G), true))

        # Check if the syndrome of the recovery operator is equal to that of the error
		@assert any(get_syndrome(S, row) == (get_syndrome(S, r2) .⊻ s2) for row in eachrow(T_0))

        println("\n", "Two level decoding procedure:\n")
        r2_twolevel, l2_twolevel, t2_twolevel, t_star2, t_s_bar2_twolevel, t_q2, t_s_C_q_bar2 = decode_EAOAQEC_two_level(s2, S, G, L, T_0, p, t_s_dict)

        println("r2_twolevel = ", print_pauli_operators(r2_twolevel, true))
        println("l2_twolevel = ", print_pauli_operators(l2, true))   
        println("t2_twolevel = ", print_pauli_operators(t2, true))
        println("t_star2 = ", print_pauli_operators(t_star2, true))
        println("t_s_bar2_twolevel = ", print_pauli_operators(t_s_bar2_twolevel, true))
        isnothing(t_q2) || println("t_q2 = ", print_pauli_operators(t_q2, true))
        isnothing(t_s_C_q_bar2) || println("t_s_C_q_bar2 = ", print_pauli_operators(t_s_C_q_bar2, true))
        println("syndrome for r2_twolevel = ", findall(!iszero, get_syndrome(S, r2_twolevel)))
        println("r2_twolevel = ", print_pauli_operators(r2_twolevel', true), ", lowest weight operator in coset: ", print_pauli_operators(lowest_weight_coset_vector(l2, t2_twolevel, S, G), true))

        # Check if the syndrome of the recovery operator is equal to that of the error
		@assert any(get_syndrome(S, row) == (get_syndrome(S, r2_twolevel) .⊻ s2) for row in eachrow(T_0))


        println("\n", "Two level decoding procedure (Type 2):\n")
        r2_two_level_2, l2_twolevel_2, t2_twolevel_2, t_star_star_star, t_s_q_bar2_twolevel_2, t_s_C_q_bar2_twolevel_2 = decode_EAOAQEC_two_level_type2(s2, S, G, L, T_0, p, t_s_dict)

        println("r2_two_level_2 = ", print_pauli_operators(r2_two_level_2, true))
        println("l2_twolevel_2 = ", print_pauli_operators(l2_twolevel_2, true))   
        println("t2_twolevel_2 = ", print_pauli_operators(t2_twolevel_2, true))
        println("t_star_star_star = ", print_pauli_operators(t_star_star_star, true))
        
        isnothing(t_s_q_bar2_twolevel_2) || println("t_s_q_bar2_twolevel_2 = ", print_pauli_operators(t_s_q_bar2_twolevel_2, true))
        isnothing(t_s_C_q_bar2_twolevel_2) || println("t_s_C_q_bar2_twolevel_2 = ", print_pauli_operators(t_s_C_q_bar2_twolevel_2, true))
        
        println("syndrome for r2_two_level_2 = ", findall(!iszero, get_syndrome(S, r2_two_level_2)))
        println("r2_two_level_2 = ", print_pauli_operators(r2_two_level_2', true), ", lowest weight operator in coset: ", print_pauli_operators(lowest_weight_coset_vector(l2_twolevel_2, t2_twolevel_2, S, G), true))

        # Check if the syndrome of the recovery operator is equal to that of the error
        @assert any(get_syndrome(S, row) == (get_syndrome(S, r2_two_level_2) .⊻ s2) for row in eachrow(T_0))

        
else
    println("missing syndromes = ", missing_syndrome)
end
end
end

main()