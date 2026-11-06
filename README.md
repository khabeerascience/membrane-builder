# membrane-builder
Build custom membrane systems using molecules that are not available in CHARMM-GUI.
Workflow for building custom membrane
Avogadro (.mol2) → CGenFF parameters → GROMACS version topology → monolayer build  → energy minimization → Extract stable frame


In this study, 18-Cys-MEA was used to build the membrane system. To evaluate the versatility of the proposed workflow, other amphiphilic molecules, such as N-palmitoyl cysteamine or other molecules of interest, can be explored by validating their structures in Avogadro, generating CGenFF parameters and GROMACS topologies, followed by monolayer construction and equilibration.


Following frame extraction, further investigations can be conducted to evaluate peptide penetration and protein stability.
