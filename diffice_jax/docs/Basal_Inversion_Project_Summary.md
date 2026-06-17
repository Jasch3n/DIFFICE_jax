## Summary of Current Status (2/12/2026)
Our goal is to add the functionality of basal inversion into the current codebase. I have already implemented a preliminary version for the pinn code (not the xpinn code), adding the "basal" option to the preprocessing, equation, and loss modules, among others. However, the code can currently only invert for either only floating or only grounded ice components at a time, but not both at the same time. 

To do so at the same time, we will need to use the xpinns, assigning different neural networks trained on different equations for floating vs. grounded regions of ice. The equations for floating ice and grounded ice have different scales, trained with different boudnary conditions and the losses are scaled differently as a consequence. 

In particular, the inversion for viscosity of grounded ice requires the boundary condition at the calving front. However, the simultaneous inversion for viscosity and basal friction of grounded ice depends requires the viscosity of ice at the floating-grounding transition as boundary condition. So these neural networks will be trained differently but simultaneously. Also, for the case of "pinning points" where the grounded region is completely surrounded by floating ice, the training for the floating network(s) cannot train at points within the grounded region over the pinning point. Are X-PINNS capable of doing this? 

Whenever we make new changes to the core diffice_jax code, these changes should be noted in Basal_Inversion_DevLog.md in the same directory. 

Whenever we want to run test scripts, make sure to use the virtual environment located at /Users/jiapchen/Research/my_diffice_jax_env.