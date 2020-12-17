#!/usr/bin/env python
# coding: utf-8

# # Model Specification
# 
# ## Model1: Varying Intercepts without Covariates
# 
# We can start our simplest model without any covariates. And random intercepts for participant pairs(PP), Region of Interest(ROI) and voxels(VOX) are considered.
# Then the lme4 format of model is:
# 
# $$\Delta_{p,r,v} \sim 1 + (1\,|\,\text{PP}) + (1\,|\,\text{ROI}) + (1\,|\,\text{VOX})$$
# 
# Corresponding model can be expressed as:
# 
# For $p=1,2,3,...,N$, $r=1,2,3,...,m$ and $v=1,2,3,...,k_{i}$:
# 
# $$
# \begin{gathered}
# \Delta_{p,r,v} \sim \text{Student t}(\nu,\mu_{p,r,v},\sigma) \\
# \mu_{p,r,v} = \alpha + \alpha_{\text{PP}_{[p]}} + \alpha_{\text{ROI}_{[r]}} + \alpha_{\text{VOX}_{[v]}} + \epsilon
# \end{gathered}
# $$
# 
# Notice that:
# 
# * $N:$ Number of participant pairs
# * $m:$ Number of Region of Interests
# * $k_{i}:$ Number of voxel in different hemispheres of insula, where $i:$ left, right;
# * $\mu_{p,r,v}:$ Mean difference of stressor response for $p^{th}$ yoked pair of participants at $v^{th}$ voxel in $r^{th}$ ROI;
# * $\alpha:$ Fixed intercept;
# * $\alpha_{\text{PP}_{[p]}}:$ Random intercept for each participant pair $p$;
# * $\alpha_{\text{ROI}_{[r]}}:$ Random intercept for each ROI $r$;
# * $\alpha_{\text{VOX}_{v}}:$ Random intercept for each voxel $v$;
# 
# ### Model1: Priors
# 
# For population level:
# 
# $$
# \begin{gathered}
# \alpha \sim \text{Student t}(3, 0, 10) \\
# \nu \sim \text{Gamma}(3.325, 0.1) \\
# \sigma \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$
# 
# For intercepts clustered by participant pairs:
# 
# $$
# \begin{gathered}
# \alpha_{\text{PP}} \sim \text{Student t}(\nu_{\text{PP}}, 0, \sigma_{\text{PP}}) \\
# \nu_{\text{PP}} \sim \text{Gamma}(3.325, 0.1) \\
# \sigma_{\text{PP}} \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$
# 
# For intercepts clustered by ROI (Marginal):
# 
# $$
# \begin{gathered}
# \alpha_{\text{ROI}} \sim \text{Student t}(\nu_{\text{ROI}}, 0, \sigma_{\alpha_{\text{ROI}}}) \\
# \nu_{\text{ROI}} \sim \text{Gamma}(3.325, 0.1) \\
# \sigma_{\alpha_{\text{ROI}}} \sim \text{HalfStudent t}(3, 0, 10) \\
# \end{gathered}
# $$
# 
# For intercepts clustered by voxel (Marginal):
# 
# $$
# \begin{gathered}
# \alpha_{\text{VOX}} \sim \text{Student t}(\nu_{\text{VOX}}, 0, \sigma_{\alpha_{\text{VOX}}}) \\
# \nu_{\text{VOX}} \sim \text{Gamma}(3.325, 0.1) \\
# \sigma_{\alpha_{\text{VOX}}} \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$

# ## Model2: Varying Intercepts with Covariates
# 
# Now we take all 5 covariates into the model. But only consider varying intercepts again. And then the model expressed in lme4 format is:
# 
# $$
# \begin{gathered}
# \text{mod} = \text{StateMean} + \text{StateDiff} + \text{TraitMean} + \text{TraitDiff} + \text{ButtonPressDiff} \\
# \Delta_{p,r,v} \sim 1 + \text{mod} + (1\,|\,\text{PP}) + (1\,|\,\text{ROI}) + (1\,|\,\text{VOX})
# \end{gathered}
# $$
# 
# Corresponding model then:
# 
# $$
# \begin{gathered}
# \Delta_{p,r,v} \sim \text{Student t}(\nu,\mu_{p,r,v},\sigma) \\
# \mu_{p,r,v} = \alpha + \alpha_{\text{PP}_{[p]}} + \alpha_{\text{ROI}_{[r]}} + \alpha_{\text{VOX}_{[v]}} + \beta\times\text{mod} + \epsilon
# \end{gathered}
# $$
# 
# Notice that:
# 
# * $\beta:$ Fixed Slope
# 
# We can also write above equation specifically as followed:
# 
# $$
# \begin{aligned}
# \mu_{p,r,v} = & \alpha + \alpha_{\text{PP}_{[p]}} + \alpha_{\text{ROI}_{[r]}} + \alpha_{\text{VOX}_{[v]}} + \\
#               & \beta_{\text{StateMean}}\text{StateMean} + \beta_{\text{StateDiff}}\text{StateDiff} + \\
#               & \beta_{\text{TraitMean}}\text{TraitMean} + \beta_{\text{TraitDiff}}\text{TraitDiff} + \\
#               & \beta_{\text{ButtonPressDiff}}\text{ButtonPressDiff} + \epsilon
# \end{aligned}
# $$
# 
# ### Model2: Priors
# 
# The priors of same coefficients are inherited from Model 1. In addition, the following priors are employed:
# 
# For population level:
# 
# $$\beta \sim \text{Student t}(3, 0, 10)$$

# ## Model3: Varying Slopes with Covariates
# 
# Next, we consider varying slopes for all 5 covariates and varying intercept for participant pair only. The lme4 format of the model is:
# 
# $$
# \Delta_{p,r,v} \sim 1 + \text{mod} + (1\,|\,\text{PP}) + (\text{mod}\,|\,\text{ROI}) + (\text{mod}\,|\,\text{VOX})
# $$
# 
# Corresponding model then:
# 
# $$
# \begin{gathered}
# \Delta_{p,r,v} \sim \text{Student t}(\nu,\mu_{p,r,v},\sigma) \\
# \mu_{p,r,v} = \alpha + \alpha_{\text{PP}_{[p]}} + (\beta + \gamma_{\text{ROI}_{[r]}} + \eta_{\text{VOX}_{[v]}})\,\text{mod} + \epsilon
# \end{gathered}
# $$
# 
# Notice that:
# 
# * $\gamma_{\text{ROI}_{[r]}}:$ Random slope for each ROI $r$;
# * $\eta_{\text{VOX}_{[v]}}:$ Random slope for each voxel $v$;
# 
# We can also write above equation specifically as followed:
# 
# $$
# \begin{aligned}
# \mu_{p,r,v} = & \alpha + \alpha_{\text{PP}_{[p]}} +\\
#               & \beta_{\text{StateMean}}\text{StateMean} + \gamma_{\text{StateMean},[r]}\text{StateMean} + \eta_{\text{StateMean},[v]}\text{StateMean} + \\
#               & \beta_{\text{StateDiff}}\text{StateDiff} + \gamma_{\text{StateDiff},[r]}\text{StateDiff} + \eta_{\text{StateDiff},[v]}\text{StateDiff} + \\
#               & \beta_{\text{TraitMean}}\text{TraitMean} + \gamma_{\text{TraitMean},[r]}\text{TraitMean} + \eta_{\text{TraitMean},[v]}\text{TraitMean} + \\
#               & \beta_{\text{TraitDiff}}\text{TraitDiff} + \gamma_{\text{TraitDiff},[r]}\text{TraitDiff} + \eta_{\text{TraitDiff},[v]}\text{TraitDiff} + \\
#               & \beta_{\text{ButtonPressDiff}}\text{ButtonPressDiff} + \gamma_{\text{ButtonPressDiff},[r]}\text{ButtonPressDiff} + \\
#               & \eta_{\text{ButtonPressDiff},[v]}\text{ButtonPressDiff} + \epsilon
# \end{aligned}
# $$
# 
# ### Model3: Priors
# 
# Besides to prior defined in Model 1 and Model 2, the following priors are also introduced:
# 
# For slopes clustered by ROI (Marginal):
# 
# $$
# \begin{gathered}
# \gamma_{\text{ROI}_{[r]}} \sim \text{Student t}(\nu_{\text{ROI}}, 0, \sigma_{\gamma_{\text{ROI}}}) \\
# \sigma_{\gamma_{\text{ROI}}} \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$
# 
# For slopes clusterd by voxel (Marginal):
# 
# $$
# \begin{gathered}
# \eta_{\text{VOX}_{[v]}} \sim \text{Student t}(\nu_{\text{VOX}}, 0, \sigma_{\eta_{\text{VOX}}}) \\
# \sigma_{\eta_{\text{Vox}}} \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$
# 

# ## Model4: Varying Intercepts and Varying Slopes with Covariates
# Finally, we take both varying intercepts and varying slopes into consideration. The lme4 format of the full model is:
# 
# $$
# \Delta_{p,r,v} \sim 1 + \text{mod} + (1\,|\,\text{PP}) + (1 + \text{mod}\,|\,\text{ROI}) + (1 + \text{mod}\,|\,\text{VOX})
# $$
# 
# Full model then:
# 
# $$
# \begin{gathered}
# \Delta_{p,r,v} \sim \text{Student t}(\nu,\mu_{p,r,v},\sigma)\\
# \mu_{p,r,v} = \alpha + \alpha_{\text{PP}_{[p]}} + \alpha_{\text{ROI}_{[r]}} + \alpha_{\text{VOX}_{[v]}} + 
#               (\beta + \gamma_{\text{ROI}_{[r]}} + \eta_{\text{VOX}_{[v]}})\,\text{mod} + \epsilon
# \end{gathered}
# $$
# 
# We can also write above equation specifically as followed:
# 
# $$
# \begin{aligned}
# \mu_{p,r,v} = & \alpha + \alpha_{\text{PP}_{[p]}} + \alpha_{\text{ROI}_{[r]}} + \alpha_{\text{VOX}_{[v]}} + \\
#               & \beta_{\text{StateMean}}\text{StateMean} + \gamma_{\text{StateMean},[r]}\text{StateMean} + \eta_{\text{StateMean},[v]}\text{StateMean} + \\
#               & \beta_{\text{StateDiff}}\text{StateDiff} + \gamma_{\text{StateDiff},[r]}\text{StateDiff} + \eta_{\text{StateDiff},[v]}\text{StateDiff} + \\
#               & \beta_{\text{TraitMean}}\text{TraitMean} + \gamma_{\text{TraitMean},[r]}\text{TraitMean} + \eta_{\text{TraitMean},[v]}\text{TraitMean} + \\
#               & \beta_{\text{TraitDiff}}\text{TraitDiff} + \gamma_{\text{TraitDiff},[r]}\text{TraitDiff} + \eta_{\text{TraitDiff},[v]}\text{TraitDiff} + \\
#               & \beta_{\text{ButtonPressDiff}}\text{ButtonPressDiff} + \gamma_{\text{ButtonPressDiff},[r]}\text{ButtonPressDiff} + \\
#               & \eta_{\text{ButtonPressDiff},[v]}\text{ButtonPressDiff} + \epsilon
# \end{aligned}
# $$
# 
# ## Model4: Priors
# 
# Here we rewrite all priors for the coefficients. Notice that now both $\alpha_{\text{ROI}_{[r]}}$, $\gamma_{\text{ROI}_{[r]}}$ and $\alpha_{\text{VOX}_{[v]}}$,
# $\gamma_{\text{VOX}_{[v]}}$ are followed in joint distributions correspondingly. Hence, two variance-covariance matrix are introduced:
# 
# For population level:
# 
# $$
# \begin{gathered}
# \alpha \sim \text{Student t}(3, 0, 10) \\
# \beta \sim \text{Student t}(3, 0, 10) \\
# \nu \sim \text{Gamma}(3.325, 0.1) \\
# \sigma \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$
# 
# For intercepts clustered by participant pairs:
# 
# $$
# \begin{gathered}
# \alpha_{\text{PP}_{[p]}} \sim \text{Student t}(\nu_{\text{PP}}, 0, \sigma_{\text{PP}}) \\
# \nu_{\text{PP}} \sim \text{Gamma}(3.325, 0.1) \\
# \sigma_{\text{PP}} \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$
# 
# For intercepts and slopes clustered by ROI (Marginal):
# 
# $$
# \begin{gathered}
# \alpha_{\text{ROI}_{[r]}} \sim \text{Student t}(\nu_{\text{ROI}}, 0, \sigma_{\alpha_{\text{ROI}}}) \\
# \gamma_{\text{ROI}_{[r]}} \sim \text{Student t}(\nu_{\text{ROI}}, 0, \sigma_{\gamma_{\text{ROI}}}) \\
# \nu_{\text{ROI}} \sim \text{Gamma}(3.325, 0.1) \\
# \sigma_{\alpha_{\text{ROI}}} \sim \text{HalfStudent t}(3, 0, 10) \\
# \sigma_{\gamma_{\text{ROI}}} \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$
# 
# For intercepts and slopes clustered by voxel (Marginal):
# 
# $$
# \begin{gathered}
# \alpha_{\text{VOX}_{[v]}} \sim \text{Student t}(\nu_{\text{VOX}}, 0, \sigma_{\alpha_{\text{VOX}}}) \\
# \eta_{\text{VOX}_{[v]}} \sim \text{Student t}(\nu_{\text{VOX}}, 0, \sigma_{\eta_{\text{VOX}}}) \\
# \nu_{\text{VOX}} \sim \text{Gamma}(3.325, 0.1) \\
# \sigma_{\alpha_{\text{VOX}}} \sim \text{HalfStudent t}(3, 0, 10) \\
# \sigma_{\eta_{\text{VOX}}} \sim \text{HalfStudent t}(3, 0, 10)
# \end{gathered}
# $$
# 
# For correlation structures in both ROI and voxel, let $j:\,\gamma_{\text{ROI}},\,\eta_{\text{VOX}}$ and $\mathbf{S}$ be the variance-covariance matrix:
# 
# $$
# \begin{gathered}
# \mathbf{\Sigma} = \begin{bmatrix}
#                   \sigma_{\alpha_{j}} &   &   &   &   &   \\
#                   &\sigma_{j_{\text{ StateMean}}} &   &   &   &   \\
#                   &   &\sigma_{j_{\text{ StateDiff}}} &    &   &   \\
#                   &   &   &\sigma_{j_{\text{ TraitMean}}}   &   &   \\
#                   &   &   &   &\sigma_{j_{\text{ TraitDiff}}}   &   \\
#                   &   &   &   &   &\sigma_{j_{\text{ ButtonPressDiff}}}
#                   \end{bmatrix} \\
# \mathbf{S} = \mathbf{\Sigma}\mathbf{R}\mathbf{\Sigma} \\
# \mathbf{R} \sim \text{LKJcorr(2)} \\
# \end{gathered}
# $$

# In[ ]:




