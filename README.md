Computational resources for "Homogenization of plasmonic crystals: Seeking the epsilon-near-zero effect"
====

This repository contains all computational resources that were used for
computations in the aforementioned publication.

All compuations were done with small C++ programs based on the deal.II
finite element library, freely available at https://www.dealii.org and
https://github.com/dealii/dealii. See
https://www.dealii.org/current/readme.html and
https://www.dealii.org/current/doxygen/deal.II/Tutorial.html for more
information about how to install deal.II and run a program based on
deal.II.

The repository structure is as follows:

loki
----

This subdirectory contains the sources for a program computing the direct
scattering problem of a plasmonic crystal. The program creates the
necessary configuration file upon first invocation.

sobek
-----

This subdirectory contains the sources for a program computing the cell
problems.

sobek-parameter-files
-----

This subdirectory contains all parameter files, as well as the
post-processed results that were used for the cell problem calculations.
