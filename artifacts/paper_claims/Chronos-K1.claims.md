# Paper Claim Audit: Chronos-K1.txt

Rule-based extraction, no LLM, no external API.

This audit does not verify proofs or replace peer review. It separates theoretical claims,
definitions, conditional claims, empirical evidence, and boundary notes while recording
`supports`, `does_not_support`, `evidence_level`, `next_gate`, and `claim_boundary`.

- total claims: 153
- by claim_type: {'boundary_note': 29, 'theoretical_claim': 80, 'conditional_claim': 20, 'definition': 14, 'empirical_evidence': 10}
- by evidence_level: {'stated_limitation': 29, 'theoretical_argument': 80, 'conditional_argument': 20, 'definitional': 14, 'numerical_experiment': 10}

## paper_0033_gravitational_content_the_thermodynamic_perspective_ab

- source: `Chronos-K1.txt:33`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: Gravitational content. The thermodynamic perspective above conditionally recovers the Einstein equations $R_{munu}-tfrac12 R\,g_{munu}=8pi T_{munu}$ via the Jacobson argument. Independently, in the spherical sector with $f>0$, the differ...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0042_the_problem_riemann_s_1854_habilitation_lecture_observ

- source: `Chronos-K1.txt:42`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: The problem. Riemann's 1854 Habilitation lecture observed that, for a continuous manifold, the basis of metric relations must be sought outside the manifold itself, in forces acting upon it cite{Riemann1854}.footnote{Riemann's formulatio...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0050_where_phi_parametrises_structural_variation_and_h_0_is

- source: `Chronos-K1.txt:50`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: where $Phi$ parametrises structural variation and $H>0$ is the local cost per unit variation. The object constrained in the present paper is not $dt_{mathrm{info}}$ itself, but the signed leading quadratic form $G$ associated with the lo...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0062_r_munu_0_theorem_ref_thm_field_the_conditions_k_i_1_ar

- source: `Chronos-K1.txt:62`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: $R_{munu}=0$ (Theorem ref{thm:field}). The conditions $K_i=1$ are
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0063_introduced_at_the_field_level_as_a_candidate_ansatz_mo

- source: `Chronos-K1.txt:63`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: introduced at the field level as a candidate ansatz, motivated by but
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0064_not_derived_from_the_point_level_condition_k_1_the_equ

- source: `Chronos-K1.txt:64`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: not derived from the point-level condition $K=1$; the equivalence is
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0066_not_a_derivation_whether_the_ansatz_admits_an_independ

- source: `Chronos-K1.txt:66`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: not a derivation. Whether the ansatz admits an independent justification
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0067_is_left_open_cf_open_problem_1

- source: `Chronos-K1.txt:67`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: is left open (cf.\ Open Problem 1).
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0073_ansatz_as_a_separate_input_neither_perspective_derives

- source: `Chronos-K1.txt:73`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: ansatz as a separate input. Neither perspective derives Einstein from
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0076_medskipnoindentmain_results_point_level_under_isotropi

- source: `Chronos-K1.txt:76`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: medskipnoindentMain results. Point-level: under isotropic linearisation, Lorentzian signature, real spectrum of the induced generator $J_G$, and the positive stability threshold $d_{c}>0$ are equivalent (Theorem ref{thm:equivalence}). Th...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0103_the_explicit_relation_is_left_to_future_work

- source: `Chronos-K1.txt:103`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: the explicit relation is left to future work.
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0114_medskipnoindentframework_definition_the_present_paper

- source: `Chronos-K1.txt:114`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: medskipnoindentFramework definition. The present paper is organised around the following layers. Input layer: a structural parameter $Phi$ and a positive scalar cost $H$, defining the information time $dt_{mathrm{info}}:=dPhi/H$. Axiom l...
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0120_where_phi_parametrises_structural_variation_and_h_0_is

- source: `Chronos-K1.txt:120`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: where $Phi$ parametrises structural variation and $H>0$ is the local scalar cost per unit variation of $Phi$ (an entropic resistance, unrelated to the matrix weight $H_{*}$ of Sref{sec:stability}). Associated with the local second-order...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0126_is_not_the_primitive_cost_functional_of_the_realizabil

- source: `Chronos-K1.txt:126`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: is not the primitive cost functional of the realizability layer, but an additional dynamical choice introduced after the Lorentzian type of $G$ has been established, in order to study local restoring behaviour near $K=1$. Hence the Theor...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0134_medskipnoindentdeeper_motivation_axioms_r_e_t_of_appen

- source: `Chronos-K1.txt:134`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: medskipnoindentDeeper motivation. Axioms R, E, T of Appendix ref{app:realizability} may be viewed as a realizability encoding of the Hessian-level degeneracy structure associated with the local second-order cost form of $dt_{mathrm{info}...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0136_medskipnoindentscope_of_the_claim_what_is_constrained

- source: `Chronos-K1.txt:136`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: medskipnoindentScope of the claim. What is constrained here is the algebraic type of $G$ (its signature), not all of its specific values. Accordingly, $G$ should be read as a local cost-form forced by realizability, rather than as a comp...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0144_medskipnoindentrole_of_information_time_in_the_present

- source: `Chronos-K1.txt:144`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: medskipnoindentRole of information time in the present paper. In the present work, $dt_{mathrm{info}}$ plays a motivating and structural role: it specifies the cost-time layer from which the local leading form $G$ is introduced. The late...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0220_noindentfrom_geometry_to_dynamics_law_i_specifies_the

- source: `Chronos-K1.txt:220`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: noindentFrom geometry to dynamics. Law I specifies the algebraic type of $G$, but not how the system evolves. The next step is to obtain from $G$, together with a conservative compatibility condition and a Euclidean normalisation convent...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0224_begin_theorem_structural_uniqueness_label_thm_unique

- source: `Chronos-K1.txt:224`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{theorem}[Structural Uniqueness]label{thm:unique}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0229_end_theorem

- source: `Chronos-K1.txt:229`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{theorem}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0236_theorem_ref_thm_unique_requires_only_det_gneq_0_it_hol

- source: `Chronos-K1.txt:236`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: Theorem ref{thm:unique} requires only $det Gneq 0$; it holds for any signature. Theorem ref{thm:spectral} provides the spectral computation; the Lorentzian condition becomes a stability criterion in Theorem ref{thm:equivalence}, under is...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0239_begin_theorem_spectral_reality_label_thm_spectral

- source: `Chronos-K1.txt:239`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{theorem}[Spectral Reality]label{thm:spectral}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0244_end_theorem

- source: `Chronos-K1.txt:244`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{theorem}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0250_begin_theorem_necessity_of_lorentzian_signature_label

- source: `Chronos-K1.txt:250`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{theorem}[Necessity of Lorentzian Signature]label{thm:necessity}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0251_if_mathrm_sig_g_2_0_or_0_2_then_det_g_0_so_lambda_2_1

- source: `Chronos-K1.txt:251`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: If $mathrm{Sig}(G)=(2,0)$ or $(0,2)$, then $det G>0$, so $lambda^2+1/det G=0$ has purely imaginary roots; no positive real $d_c>0$ exists in the sense of Theorem ref{thm:equivalence}.
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0252_end_theorem

- source: `Chronos-K1.txt:252`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{theorem}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0275_proposition_ref_prop_diag_inv_but_the_full_lyapunov_st

- source: `Chronos-K1.txt:275`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: (Proposition ref{prop:diag_inv}), but the full Lyapunov stability
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0277_stated_in_theorem_ref_thm_equivalence_therefore_holds

- source: `Chronos-K1.txt:277`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: stated in Theorem ref{thm:equivalence} therefore holds emph{within
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0283_begin_theorem_signature_stability_equivalence_label_th

- source: `Chronos-K1.txt:283`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{theorem}[Signature-Stability Equivalence]label{thm:equivalence}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0293_for_general_h_the_zero_eigenvalue_locus_in_d_is_invari

- source: `Chronos-K1.txt:293`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: For general $H_{*}$, the zero-eigenvalue locus in $d$ is invariant under invertible right multiplication by $H_{*}$ (Proposition ref{prop:diag_inv}), but the full real-part stability interpretation may fail (Remark ref{rem:Hstar_dep}); t...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0294_end_theorem

- source: `Chronos-K1.txt:294`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{theorem}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0305_h_scalar_sref_sec_lawi_is_the_local_cost_per_unit_vari

- source: `Chronos-K1.txt:305`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: $H$ (scalar, Sref{sec:lawI}) is the local cost per unit variation of $Phi$ (an entropic resistance), appearing in $dt_{mathrm{info}}=dPhi/H$; $H_{*}$ (matrix) denotes the cost-function weight matrix in $d(delta x)/dt=(J_{G}-dI)H_{*}\,del...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0309_with_g_mathrm_diag_0_1_10_and_h_bigl_begin_smallmatrix

- source: `Chronos-K1.txt:309`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: With $G=mathrm{diag}(0.1,-10)$ and $H_{*}=bigl(begin{smallmatrix}1&0.9\\0.9&1end{smallmatrix}bigr)$, $d_{c}^{mathrm{iso}}=1$ while $d_{c}(H_{*})=4.55$, confirming that $d_{c}$ may depend on $H_{*}$ outside the isotropic class. Here $d_{c...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0312_begin_proposition_h_invariance_of_the_zero_eigenvalue

- source: `Chronos-K1.txt:312`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{proposition}[$H_{*}$-invariance of the zero-eigenvalue locus]label{prop:diag_inv}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0314_end_proposition

- source: `Chronos-K1.txt:314`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{proposition}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0319_for_the_rindler_metric_with_diagonal_h_0_the_zero_eige

- source: `Chronos-K1.txt:319`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: For the Rindler metric with diagonal $H_{*}>0$, the zero-eigenvalue threshold in $d$ remains $d_{c}=alpha/(kappaell)$. The thermodynamic applications of S ref{sec:thermo} use this threshold directly, without requiring the full isotropic...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0337_draw_equiv_core_node_label_above_right_2pt_sloped_d_c

- source: `Chronos-K1.txt:337`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: draw[equiv] (core) -- node[label, above right=-2pt, sloped] {$d_{c}$ definition} (dc);
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_0345_caption_signature_spectrum_stability_triple_correspond

- source: `Chronos-K1.txt:345`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: caption{Signature-spectrum-stability triple correspondence. The core algebraic sign $det G<0$ corresponds, under 2D nondegeneracy, to three expressions: Lorentzian signature $mathrm{Sig}(G)=(1,1)$ (geometric, by Sylvester's inertia law);...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0353_noindentlaw_ii_textit_the_generator_of_the_dynamics_sp

- source: `Chronos-K1.txt:353`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: noindentLaw II. textit{The generator of the dynamics splits as $GA = alpha J - GD$ with $Dge 0$ (positive-semidefinite dissipation). This symplectic-dissipative decomposition is the content of Law II; it is a structural axiom, not derive...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0359_the_splitting_ga_alpha_j_gd_is_the_axiomatic_content_o

- source: `Chronos-K1.txt:359`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The splitting $GA=alpha J-GD$ is the axiomatic content of Law II. Combined with Theorem ref{thm:unique} (which fixes $J_{G}$), this yields the unique admissible evolution law (Theorem ref{thm:lawII}).
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0361_begin_theorem_canonical_form_of_law_ii_label_thm_lawii

- source: `Chronos-K1.txt:361`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{theorem}[Canonical Form of Law II]label{thm:lawII}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0362_let_g_be_non_degenerate_and_let_j_g_alpha_g_1_j_textup

- source: `Chronos-K1.txt:362`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: Let $G$ be non-degenerate and let $J_{G}=alpha G^{-1}J$ textup{(Theorem ref{thm:unique})}. For any generator satisfying the Law II constraint
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0371_end_theorem

- source: `Chronos-K1.txt:371`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{theorem}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0422_theorem_ref_thm_restoring_and_second_the_natural_separ

- source: `Chronos-K1.txt:422`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: (Theorem ref{thm:restoring}); and second, the natural separation of
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0428_begin_theorem_local_effective_restoring_rate_label_thm

- source: `Chronos-K1.txt:428`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{theorem}[Local Effective Restoring Rate]label{thm:restoring}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0429_let_g_be_a_2times_2_lorentzian_metric_d_d_c_i_law_iii

- source: `Chronos-K1.txt:429`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: Let $G$ be a $2times 2$ Lorentzian metric, $D=d_{c}I$ (Law III), and $J_{G}=alpha G^{-1}J$ (Theorem ref{thm:unique}). Define $K(x)=x^{top}Gx$, $V=tfrac{1}{2}(K-1)^{2}$, with drift $dot{x}=(J_{G}-d_{c}I)nabla V$. Then for all $xinmathbb{R...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0441_end_theorem

- source: `Chronos-K1.txt:441`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{theorem}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0498_with_this_choice_theorem_ref_thm_restoring_shows_that

- source: `Chronos-K1.txt:498`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: With this choice, Theorem ref{thm:restoring} shows that $K{=}1$ is a
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0539_x_top_g_2_x_1_cf_theorem_ref_thm_restoring_on

- source: `Chronos-K1.txt:539`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: $x_{*}^{top}G^{2}x_{*}=1$ (cf.\ Theorem ref{thm:restoring}). On
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0556_proposition_ref_prop_recovery_the_single_leaf_sigma_0

- source: `Chronos-K1.txt:556`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: (Proposition ref{prop:recovery}); the single leaf $Sigma_0=\{c=0\}$ is
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0565_at_rate_4c_2_proposition_ref_prop_recovery_failing_to

- source: `Chronos-K1.txt:565`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: at rate $4c^2$, Proposition ref{prop:recovery}), failing to attract only on
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0570_is_taken_up_in_open_problem_6_sref_sec_discussion_for

- source: `Chronos-K1.txt:570`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: is taken up in Open Problem 6 (Sref{sec:discussion}); for a general
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0574_begin_remark_numerical_diagnostic

- source: `Chronos-K1.txt:574`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: begin{remark}[Numerical diagnostic]
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_0585_law_iii_rather_than_a_numerical_artifact

- source: `Chronos-K1.txt:585`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: Law III rather than a numerical artifact.
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_0588_begin_proposition_exact_leaf_dynamics_and_recovery_tim

- source: `Chronos-K1.txt:588`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{proposition}[Exact leaf dynamics and recovery-time scaling]label{prop:recovery}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0613_end_proposition

- source: `Chronos-K1.txt:613`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{proposition}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0615_begin_remark_exact_leaf_numerical_reproduction_of_the

- source: `Chronos-K1.txt:615`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: begin{remark}[Exact-leaf numerical reproduction of the closed-form recovery law]
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_0616_direct_integration_on_exact_invariant_leaves_reproduce

- source: `Chronos-K1.txt:616`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: Direct integration on exact invariant leaves reproduces the closed-form
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_0617_recovery_law_of_proposition_ref_prop_recovery_to_solve

- source: `Chronos-K1.txt:617`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: recovery law of Proposition ref{prop:recovery} to solver accuracy. With
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0618_exact_on_leaf_initialisation_x_1_x_2_c_held_exactly_th

- source: `Chronos-K1.txt:618`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: exact on-leaf initialisation ($x_1+x_2=c$ held exactly), the measured
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_0636_approached_proposition_ref_prop_recovery

- source: `Chronos-K1.txt:636`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: approached (Proposition ref{prop:recovery}).
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0641_this_section_develops_the_thermodynamic_perspective_as

- source: `Chronos-K1.txt:641`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: This section develops the thermodynamic perspective as a conditional bridge to the Einstein equations. Every external identification used is marked explicitly; the full input list is summarised at the end of the section.
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0663_textup_external_g_is_the_standard_rindler_metric_from

- source: `Chronos-K1.txt:663`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: textup{[external: $g$ is the standard Rindler metric from GR, not derived from Law I]},
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0673_in_this_subsection_we_work_within_the_sekimoto_seifert

- source: `Chronos-K1.txt:673`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: In this subsection we work within the Sekimoto--Seifert stochastic thermodynamics framework cite{Sekimoto}; within that external framework, the key additional external input is the identification $T_{mathrm{eff}}=T_{mathrm{tol}}$, which...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0675_medskipnoindentscope_of_the_ou_reduction_the_ou_reduct

- source: `Chronos-K1.txt:675`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: medskipnoindentScope of the OU reduction. The OU reduction below presupposes a strictly positive restoring rate toward $\{K=1\}$. It therefore applies wherever that rate is strictly positive, and degenerates where it vanishes --- namely...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0686_consistent_with_law_ii_and_theorem_ref_thm_restoring_t

- source: `Chronos-K1.txt:686`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: consistent with Law II and Theorem ref{thm:restoring}. This is, at this stage, a formal effective-temperature definition: its identification with a physical horizon temperature requires the external input $T_{mathrm{eff}}=T_{mathrm{tol}}...
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0690_the_heat_definition_follows_the_sekimoto_seifert_stoch

- source: `Chronos-K1.txt:690`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: (The heat definition follows the Sekimoto--Seifert stochastic thermodynamics convention cite{Sekimoto}.) A derivation of the identification $T_{mathrm{eff}}=T_{mathrm{tol}}$ from a more fundamental principle (e.g.\ a KMS condition cite{H...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0693_the_following_statement_is_conditional_on_the_external

- source: `Chronos-K1.txt:693`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: The following statement is conditional on the external identification $T_{mathrm{eff}}=T_{mathrm{tol}}$; it is not an autonomous derivation of the Clausius relation, but the equivalence theorem establishing what $sigma^{2}$ must be when...
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0695_begin_proposition_local_clausius_equivalence_label_pro

- source: `Chronos-K1.txt:695`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{proposition}[Local Clausius Equivalence]label{prop:clausius}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0700_end_proposition

- source: `Chronos-K1.txt:700`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{proposition}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0707_begin_remark_assumptions_of_proposition_ref_prop_claus

- source: `Chronos-K1.txt:707`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{remark}[Assumptions of Proposition ref{prop:clausius}]
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0708_the_assumptions_entering_proposition_ref_prop_clausius

- source: `Chronos-K1.txt:708`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The assumptions entering Proposition ref{prop:clausius} are:
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0730_since_every_local_rindler_horizon_admits_the_standard

- source: `Chronos-K1.txt:730`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: Since every local Rindler horizon admits the standard Rindler approximation cite{Jacobson} with $g=mathrm{diag}(-(kappaell)^{2},1)$ (where $kappa$ and $ell$ vary by location), Proposition ref{prop:clausius} applies to each such horizon,...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0735_the_thermodynamic_perspective_is_a_conditional_bridge

- source: `Chronos-K1.txt:735`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: The thermodynamic perspective is a conditional bridge rather than an autonomous derivation of the Einstein equations. Its internal inputs are the Lorentzian structure supplied by Law I, the isotropic stability scale $d_{c}$, and the OU d...
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0740_we_now_consider_at_the_field_level_the_differential_co

- source: `Chronos-K1.txt:740`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: We now consider, at the field level, the differential conditions $K_1=K_2=1$, where $K_i:=sigma_2^2Boxlnsigma_i$ are constructed from the symplectic eigenvalues $sigma_1:=rsqrt{f}$ and $sigma_2:=r$ of the spherically symmetric metric. Th...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0754_begin_definition_eigenvalue_self_consistency_functiona

- source: `Chronos-K1.txt:754`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: begin{definition}[Eigenvalue self-consistency functionals]label{def:esc}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_0759_end_definition

- source: `Chronos-K1.txt:759`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: end{definition}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_0764_these_conditions_as_a_candidate_field_level_ansatz_whe

- source: `Chronos-K1.txt:764`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: these conditions as a candidate field-level ansatz; whether they admit
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0766_problem_1_below_the_flavour_of_the_ansatz_is_similar_t

- source: `Chronos-K1.txt:766`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: Problem 1 below). The flavour of the ansatz is similar to the point-level
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0769_are_not_derived_from_the_point_level_ones

- source: `Chronos-K1.txt:769`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: are not derived from the point-level ones.
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0781_begin_lemma_scalar_curvature_formula_label_lem_r

- source: `Chronos-K1.txt:781`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{lemma}[Scalar Curvature Formula]label{lem:R}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0783_end_lemma

- source: `Chronos-K1.txt:783`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{lemma}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0791_the_following_theorem_establishes_an_algebraic_equival

- source: `Chronos-K1.txt:791`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: The following theorem establishes an algebraic equivalence between the candidate ansatz and the vacuum Einstein equations in the spherical sector. It is a reformulation of vacuum Einstein in cost-structural language, conditional on adopt...
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0793_begin_theorem_algebraic_equivalence_label_thm_field

- source: `Chronos-K1.txt:793`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{theorem}[Algebraic Equivalence]label{thm:field}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0794_for_eqref_eq_metric_with_fin_c_2_i_and_f_0_on_an_inter

- source: `Chronos-K1.txt:794`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: For eqref{eq:metric} with $fin C^{2}(I)$ and $f>0$ on an interval $Isubset(0,infty)$ (so that $sigma_{1}=rsqrt{f}$ and $lnsigma_{1}$ are real), and $K_1,K_2$ as in Definition ref{def:esc}:
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_0800_end_theorem

- source: `Chronos-K1.txt:800`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{theorem}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0808_the_trace_gives_r_g_munu_r_munu_0_by_lemma_ref_lem_r_r

- source: `Chronos-K1.txt:808`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The trace gives $R=g^{munu}R_{munu}=0$; by Lemma ref{lem:R}, $R=2(1-K_1)/r^2$, hence $K_1=1$.
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0812_begin_corollary

- source: `Chronos-K1.txt:812`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{corollary}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0816_recovering_the_schwarzschild_solution_uniquely_in_agre

- source: `Chronos-K1.txt:816`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: recovering the Schwarzschild solution uniquely, in agreement with Birkhoff's theorem.
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0821_end_corollary

- source: `Chronos-K1.txt:821`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{corollary}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0823_begin_lemma_asymmetric_rigidity_label_lem_rigidity

- source: `Chronos-K1.txt:823`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{lemma}[Asymmetric rigidity]label{lem:rigidity}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0827_item_textup_b_k_1_equiv_1_on_i_if_and_only_if_f_1_a_r

- source: `Chronos-K1.txt:827`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: item[textup{(b)}] $K_{1}equiv 1$ on $I$ if and only if $f=1+A/r+B/r^{2}$ for some constants $A,B$. Substituting into the definition of $K_{2}$ gives
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_0831_so_the_converse_of_a_fails_when_bneq_0_k_1_equiv_1_doe

- source: `Chronos-K1.txt:831`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: so the converse of (a) fails when $Bneq 0$: $K_{1}equiv 1$ does not imply $K_{2}equiv 1$.
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0834_end_lemma

- source: `Chronos-K1.txt:834`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{lemma}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0842_lemma_ref_lem_rigidity_formalises_the_asymmetry_k_2_eq

- source: `Chronos-K1.txt:842`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: Lemma ref{lem:rigidity} formalises the asymmetry: $K_{2}equiv 1$ is rigid (forces $K_{1}equiv 1$ via $f=1+C/r$), whereas $K_{1}equiv 1$ is flexible (admits a one-parameter family of $K_{2}$-violating solutions). This asymmetry underlies...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0849_begin_corollary_lambda_extension_of_theorem_ref_thm_fi

- source: `Chronos-K1.txt:849`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{corollary}[$Lambda$-extension of Theorem ref{thm:field}]label{cor:SdS}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0850_for_metric_eqref_eq_metric_with_fin_c_2_i_and_f_0_on_i

- source: `Chronos-K1.txt:850`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: For metric eqref{eq:metric} with $fin C^{2}(I)$ and $f>0$ on $Isubset(0,infty)$, and $K_{1},K_{2}$ as in Definition ref{def:esc}:
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_0856_end_corollary

- source: `Chronos-K1.txt:856`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{corollary}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0869_propagate_measured_per_unit_area

- source: `Chronos-K1.txt:869`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: propagate, measured per unit area.
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_0902_the_reader_may_wonder_whether_theorem_ref_thm_field_co

- source: `Chronos-K1.txt:902`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The reader may wonder whether Theorem ref{thm:field} constitutes a
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0904_the_theorem_establishes_an_algebraic_equivalence_betwe

- source: `Chronos-K1.txt:904`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The theorem establishes an algebraic equivalence between two systems
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0907_as_an_ansatz_the_framework_s_axioms_laws_i_iii_constra

- source: `Chronos-K1.txt:907`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: as an ansatz; the framework's axioms (Laws I--III) constrain the
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0914_as_a_field_level_ansatz_then_the_vacuum_einstein_equat

- source: `Chronos-K1.txt:914`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: as a field-level ansatz, then the vacuum Einstein equations admit an
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_0916_reformulation_not_a_derivation_whether_this_particular

- source: `Chronos-K1.txt:916`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: reformulation, not a derivation. Whether this particular ansatz is
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0918_beyond_the_algebraic_equivalence_remains_open_cf_open

- source: `Chronos-K1.txt:918`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: beyond the algebraic equivalence --- remains open (cf.\ Open Problem 1).
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0927_remain_an_independent_ansatz_cf_open_problem_1_no_logi

- source: `Chronos-K1.txt:927`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: remain an independent ansatz (cf.\ Open Problem 1). No logical dependence
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0936_begin_proposition_positivity_from_axiom_t_label_prop_s

- source: `Chronos-K1.txt:936`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{proposition}[Positivity from Axiom T]label{prop:sigma1}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0938_end_proposition

- source: `Chronos-K1.txt:938`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{proposition}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0950_begin_proposition_self_consistency_resolution_scale_la

- source: `Chronos-K1.txt:950`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{proposition}[Self-consistency resolution scale]label{prop:domain}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0952_end_proposition

- source: `Chronos-K1.txt:952`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{proposition}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_0973_the_derivation_of_sigma_1_c_makes_explicit_use_of_the

- source: `Chronos-K1.txt:973`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: The derivation of $sigma_{1,c}$ makes explicit use of the external identification $T_{mathrm{eff}}=T_{mathrm{tol}}$, which belongs to the thermodynamic perspective (Sref{sec:thermo}). Within the algebraic perspective, where no such therm...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0984_within_the_thermodynamic_perspective_once_the_required

- source: `Chronos-K1.txt:984`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: Within the thermodynamic perspective, once the required external identifications are supplied, the Einstein equations are conditionally recovered via stochastic dynamics and the Jacobson argument (Sref{sec:thermo}). The algebraic perspec...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_0986_the_input_structure_of_both_perspectives_including_the

- source: `Chronos-K1.txt:986`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The input structure of both perspectives --- including the Jacobson inputs and the external identifications $T_{mathrm{eff}}=T_{mathrm{tol}}$ and $Spropto A$ --- is summarised in Fig. ref{fig:routes} and detailed in Sref{sec:thermo}. The...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1029_draw_conjarr_k1_to_bend_left_10_node_right_1pt_font_ti

- source: `Chronos-K1.txt:1029`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: draw[conjarr] (K1) to[bend left=10] node[right=1pt, font=tinyitshape, text=black!60] {field-level ansatz (motivated by)} (sigma);
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1030_draw_biarr_sigma_node_right_font_tinyitshape_text_blac

- source: `Chronos-K1.txt:1030`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: draw[biarr] (sigma) -- node[right, font=tinyitshape, text=black!60, align=left] {under this ansatz,\algebraic equivalence\\(spherical vacuum)} (vac);
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1036_caption_two_complementary_perspectives_connecting_to_t

- source: `Chronos-K1.txt:1036`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: caption{Two complementary perspectives connecting to the Einstein equations. Thermodynamic perspective (left): Class A fluctuations near $\{K=1\}$ $to$ OU dynamics $to$ local Clausius $to$ Jacobson argument $to$ Einstein equations. The J...
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_1042_the_thermodynamic_perspective_given_the_external_ident

- source: `Chronos-K1.txt:1042`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: The thermodynamic perspective, given the external identifications $T_{mathrm{eff}}=T_{mathrm{tol}}$ and $Spropto A$, conditionally recovers the Einstein equations $R_{munu}-tfrac12 R\,g_{munu}=8pi T_{munu}$ through the Jacobson argument....
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1044_the_contribution_of_the_present_paper_is_therefore_con

- source: `Chronos-K1.txt:1044`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The contribution of the present paper is therefore conceptual and structural: Lorentzian signature is supplied internally by the cost-realizability bridge of Law I, rather than assumed as a kinematic primitive; Theorem ref{thm:equivalenc...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1046_accordingly_the_matter_density_rho_1_k_2_8pi_r_2_and_t

- source: `Chronos-K1.txt:1046`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: Accordingly, the matter density $rho=(1-K_2)/(8pi r^2)$ and the interpretation of matter are GR rewritings; $Lambda$ is not predicted by the framework; $Spropto A$ remains an external input borrowed from black-hole thermodynamics. The al...
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1050_the_following_remarks_are_heuristic_and_should_be_read

- source: `Chronos-K1.txt:1050`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The following remarks are heuristic and should be read as an outlook rather than a partial extension theorem. In $(1{+}3)$ dimensions, $\{K{=}1\}cong H^3$ carries $S^2$ cross-sections with area $4pisigma_1^2$ when $alpha=1$ and $sigma_1=...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1055_item_the_algebraic_perspective_is_currently_confined_t

- source: `Chronos-K1.txt:1055`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: item The algebraic perspective is currently confined to spherical symmetry, and depends on the field-level ansatz $K_i=1$ as a separate input not derived from Laws I--III. (scope and ansatz status)
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: record boundary and prevent overclaiming
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_1065_the_first_three_problems_concern_structural_completion

- source: `Chronos-K1.txt:1065`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: The first three problems concern structural completion and empirical distinctness; the last three concern internal dynamical and quantum consistency.
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_1077_to_and_decides_the_status_of_the_conditional_beta_r_3

- source: `Chronos-K1.txt:1077`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: to, and decides the status of, the conditional $beta/r^3$ correction
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1079_that_correction_from_conditional_to_predicted_whereas

- source: `Chronos-K1.txt:1079`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: that correction from conditional to predicted, whereas $eta=1$ removes
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1084_item_identify_testable_strong_field_predictions_within

- source: `Chronos-K1.txt:1084`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: item Identify testable strong-field predictions. Within the present spherical-vacuum analysis, since in the weak-field regime $K_2=1$ already reproduces the Newtonian potential while $K_1=1$ is automatic at linear order, any observationa...
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_1087_item_for_the_canonical_form_the_critically_damped_gene

- source: `Chronos-K1.txt:1087`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: item For the canonical form, the critically damped generator $A_{c}=J_{G}-d_{c}I$ is rank-one and admits the first integral $c=x_1+x_2$, whose foliation contains the degenerate freeze leaf $Sigma_0=\{c=0\}$ on which $K|_{Sigma_0}=0$ (Sre...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1095_this_appendix_establishes_in_self_contained_form_the_r

- source: `Chronos-K1.txt:1095`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: This appendix establishes, in self-contained form, the result invoked throughout the main text: under three explicit axioms on a primitive cost function $d$, the leading quadratic form $Q$ extracted from $d^{2}$ is forced to have Lorentz...
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified

## paper_1103_begin_definition_axiom_r_zero_threshold_realizability

- source: `Chronos-K1.txt:1103`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: begin{definition}[Axiom R --- Zero-threshold realizability]label{app:def:R}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_1109_end_definition

- source: `Chronos-K1.txt:1109`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: end{definition}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_1113_begin_definition_axiom_e_quadratic_expansion_label_app

- source: `Chronos-K1.txt:1113`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: begin{definition}[Axiom E --- Quadratic expansion]label{app:def:E}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_1120_end_definition

- source: `Chronos-K1.txt:1120`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: end{definition}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_1122_begin_definition_axiom_t_temporal_cost_label_app_def_t

- source: `Chronos-K1.txt:1122`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: begin{definition}[Axiom T --- Temporal cost]label{app:def:T}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_1127_end_definition

- source: `Chronos-K1.txt:1127`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: end{definition}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_1130_the_logical_structure_is_that_d_is_primitive_and_q_is

- source: `Chronos-K1.txt:1130`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: The logical structure is that $d$ is primitive and $Q$ is derived. Axioms R and T constrain $d$ directly, while Axiom E extracts from $d^{2}$ a quadratic leading-order form $Q$. This is what makes the Lorentzian signature theorem nontriv...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1135_begin_lemma_null_sign_equivalence_label_app_lem_null

- source: `Chronos-K1.txt:1135`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{lemma}[Null--sign equivalence]label{app:lem:null}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1137_end_lemma

- source: `Chronos-K1.txt:1137`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{lemma}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1151_begin_theorem_lorentzian_signature_from_realizability

- source: `Chronos-K1.txt:1151`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: begin{theorem}[Lorentzian signature from realizability]label{app:thm:lorentz}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1157_end_theorem

- source: `Chronos-K1.txt:1157`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: end{theorem}
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1160_step_1_null_direction_from_r_by_axiom_r_there_exists_a

- source: `Chronos-K1.txt:1160`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: Step 1 (Null direction from R). By Axiom R, there exists a subsequence $v_{n_{k}}to vin S^{1}$ with $v_{t}>0$, $v_{r}neq 0$. By Lemma ref{app:lem:null}, $Q(v)le 0$.
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1168_since_v_r_neq_0_the_denominator_is_nonzero_for_all_sin

- source: `Chronos-K1.txt:1168`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: Since $v_{r}neq 0$, the denominator is nonzero for all $sin[0,1]$, so $gamma$ is well-defined. Since $v_{t}>0$, the numerator has strictly positive $t$-component throughout, so $gamma$ is a continuous path in $S^{1}cap\{u_{t}>0\}$. Then...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1176_theorem_ref_app_thm_lorentz_takes_nondegeneracy_of_q_a

- source: `Chronos-K1.txt:1176`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: Theorem ref{app:thm:lorentz} takes nondegeneracy of $Q$ as an explicit hypothesis. Ruling out degeneracy from Axioms R, E, T alone is not possible. Consider
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1188_under_the_hypotheses_of_theorem_ref_app_thm_lorentz_ev

- source: `Chronos-K1.txt:1188`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: Under the hypotheses of Theorem ref{app:thm:lorentz}, every non-Lorentzian signature is excluded:
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1190_item_i_positive_definite_signature_2_0_is_excluded_by

- source: `Chronos-K1.txt:1190`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: item[(i)] Positive-definite signature $(2,0)$ is excluded by Axiom R together with Lemma ref{app:lem:null}: Axiom R yields a nonzero limiting direction $v$ with $v_{t}>0$ and $Q(v)le 0$, which is impossible for a positive-definite form.
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1199_in_the_body_of_the_paper_the_leading_quadratic_form_g

- source: `Chronos-K1.txt:1199`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: In the body of the paper, the leading quadratic form $G$ of Law I (Sref{sec:lawI}) is identified with the form $Q$ of this appendix. Axioms R, E, T are imposed on the local cost-form layer of the information-time construction $dt_{mathrm...
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1216_theorem_ref_thm_field_this_appendix_examines_a_relaxed

- source: `Chronos-K1.txt:1216`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: (Theorem ref{thm:field}). This appendix examines a relaxed variational
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1218_conditional_non_gr_correction_term_that_the_relaxation

- source: `Chronos-K1.txt:1218`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: conditional non-GR correction term that the relaxation produces. The
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1232_with_k_1_k_2_as_in_definition_ref_def_esc_the_two_func

- source: `Chronos-K1.txt:1232`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: with $K_1,K_2$ as in Definition ref{def:esc}. The two functionals are
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_1233_not_independent_a_direct_computation_from_definition_r

- source: `Chronos-K1.txt:1233`
- claim_type: `definition`
- evidence_level: `definitional`
- allowed_action: `record`
- claim_text_preview: not independent: a direct computation from Definition ref{def:esc}
- supports: paper introduces terminology or a formal definition
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; empirical or theoretical validation by itself
- next_gate: use the definition in downstream claims
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as definition, not independently verified

## paper_1267_conditional_statement_it_holds_only_given_the_criterio

- source: `Chronos-K1.txt:1267`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: conditional statement: it holds only given the criterion
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1314_the_beta_r_3_correction_of_eqref_eq_fform_remains_cond

- source: `Chronos-K1.txt:1314`
- claim_type: `conditional_claim`
- evidence_level: `conditional_argument`
- allowed_action: `record`
- claim_text_preview: the $beta/r^3$ correction of eqref{eq:fform} --- remains conditional on
- supports: paper states a conditional result given explicit inputs or ansatz
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; unconditional result; depends on stated external input or ansatz not derived here
- next_gate: discharge or independently justify the external inputs or ansatz
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as conditional_claim, not independently verified

## paper_1315_an_external_choice_and_is_not_a_theorem_of_the_present

- source: `Chronos-K1.txt:1315`
- claim_type: `theoretical_claim`
- evidence_level: `theoretical_argument`
- allowed_action: `record`
- claim_text_preview: an external choice and is not a theorem of the present $K{=}1$ framework.
- supports: paper states a theoretical result under manuscript assumptions
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; machine-checked proof
- next_gate: verify proof details and connect to reproducible checks
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as theoretical_claim, not independently verified

## paper_1319_sharpened_form_of_open_problem_1

- source: `Chronos-K1.txt:1319`
- claim_type: `boundary_note`
- evidence_level: `stated_limitation`
- allowed_action: `record`
- claim_text_preview: sharpened form of Open Problem 1.
- supports: paper records a limitation, scope boundary, or non-claim
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; positive scientific claim beyond the recorded boundary; claim limited by explicit manuscript boundary wording
- next_gate: resolve the explicitly stated open problem
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as boundary_note, not independently verified

## paper_1330_medskipnoindentdata_availability_no_primary_research_d

- source: `Chronos-K1.txt:1330`
- claim_type: `empirical_evidence`
- evidence_level: `numerical_experiment`
- allowed_action: `record`
- claim_text_preview: medskipnoindentData availability. No primary research datasets were generated during the current study. The symbolic and numerical verification scripts that reproduce the recovery-time scaling of Proposition ref{prop:recovery} and the al...
- supports: paper reports numerical, simulation, or experimental evidence
- does_not_support: proof verification by this audit; peer-review replacement; experimental validation; universal physics AI; formal proof or certified mechanism
- next_gate: replicate evidence and test robustness against confounds
- claim_boundary: rule-based paper audit; valid only as extraction under manuscript wording; extracted as empirical_evidence, not independently verified
