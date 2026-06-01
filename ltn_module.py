import torch

# ==========================================================
# 1. REAL LOGIC OPERATORS (Łukasiewicz Semantics)
# ==========================================================

def ltn_NOT(t):
    """G(¬φ) = 1 - G(φ)"""
    # This remains the same as classical negation
    return 1.0 - t

def ltn_AND(t1, t2):
    """G(φ ∧ ψ) = max(0, G(φ) + G(ψ) - 1)"""
    # Note: torch.clamp(x, min=0) implements max(0, x)
    return torch.clamp(t1 + t2 - 1.0, min=0.0)

def ltn_OR(t1, t2):
    """G(φ ∨ ψ) = min(1, G(φ) + G(ψ))"""
    # Note: torch.clamp(x, max=1) implements min(1, x)
    return torch.clamp(t1 + t2, max=1.0)

def ltn_IMPLIES(t_A, t_B):
    """G(A → B) = G(¬A ∨ B) = min(1, 1 - G(A) + G(B))"""
    # Implemented directly using the composite rule
    return torch.clamp(1.0 - t_A + t_B, max=1.0)

# ==========================================================
# 2. QUANTIFIER
# ==========================================================

def ltn_FORALL(truth_values_list, p=2.0):
    """
    Implements the Universal Quantifier using the Generalized Mean (smooth min).
    G(∀x φ(x)) ≈ 1 - ( 1/n * Sum((1 - G(φ(xi)))^p) )^(1/p)
    """
    if truth_values_list.numel() == 0:
        return torch.tensor(1.0, device=truth_values_list.device)
        
    # Clamping is necessary for robustness against float precision near 0/1
    one_minus_phi = ltn_NOT(truth_values_list).clamp(min=1e-9, max=1.0)
    
    # Inner sum calculation
    inner_sum = torch.mean(one_minus_phi.pow(p)) 
    
    # Outer term calculation
    outer_term = torch.pow(inner_sum.clamp(min=1e-9), 1.0 / p)
    
    # Final result
    return 1.0 - outer_term

# ==========================================================
# 3. LTN LOSS FUNCTION
# ==========================================================

def ltn_canonical_loss(logits_flat, env_flat, class_names, rules, p_agg=2.0):
    """
    Calculates the canonical LTN loss: L_LTN = Sum_axioms (1 - G(Axiom)).
    Axiom form: ∀x, isZone_j(x) → ¬isLithofacies_i(x)
    """
    if env_flat.ndim == 2:
        env_flat = env_flat.squeeze(-1)
    
    probs_all = torch.softmax(logits_flat, dim=1) 
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    
    total_axiom_truth = []
    unique_envs = torch.unique(env_flat)
    
    for env_id in unique_envs:
        env_key = str(env_id.item())
        if env_key not in rules:
            continue
            
        mask = (env_flat == env_id)
        if mask.sum() < 2:
            continue
            
        probs_j = probs_all[mask] # Samples belonging to Environment j
        
        # Antecedent G(A): isZone_j(x). G(A) = 1.0 for these masked samples.
        G_A = torch.ones(probs_j.shape[0], device=probs_j.device) 
        
        allowed = set(rules[env_key]["allowed_facies"])
        disallowed_facies = [cname for cname in class_names if cname not in allowed]
        
        for facies_name in disallowed_facies:
            idx_i = name_to_idx[facies_name]
            
            # Consequent Grounding: G(¬isLithofacies_i(x))
            G_isLitho_i = probs_j[:, idx_i].clamp(min=1e-9, max=1.0) 
            G_B = ltn_NOT(G_isLitho_i) 
            
            # Implication Grounding: G(A → B)
            G_A_implies_B = ltn_IMPLIES(G_A, G_B) 
            
            # Universal Quantifier: G(∀x (A → B))
            G_Axiom = ltn_FORALL(G_A_implies_B, p=p_agg)
            
            total_axiom_truth.append(G_Axiom)

    if not total_axiom_truth:
        return torch.tensor(0.0, device=logits_flat.device)

    # LTN Loss: Sum of (1 - G(Axiom))
    ltn_loss = torch.sum(ltn_NOT(torch.stack(total_axiom_truth)))
    
    # Return mean loss per axiom
    return ltn_loss / len(total_axiom_truth)