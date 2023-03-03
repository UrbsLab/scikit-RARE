scikit-RARE
======================================


**RARE: Relevant Association Rare-variant-bin Evolver** is an evolutionary algorithm approach to binning rare variants as a rare variant association analysis tool. scikit-RARE is scikit compatible pypi package for the RARE algotithm.

RARE constructs bins of rare variant features with relevant association to class (univariate and/or multivariate interactions) through the following steps:

1. Random bin initializaiton or expert knowledge input
2. Repeated evolutionary cycles consisting of:
   - Candidate bin evaluation with univariate scoring (chi-square test) or Relief-based scoring (MultiSURF algorithm); note: new scoring options currently under testing
   - Genetic operations (parent selection, crossover, and mutation) to generate the next generation of candidate bins
3. Final bin evaluation and summary of top bins


Installation
-----------------------------

We can easily install scikit-rare using the following command:

.. code-block:: bash

    pip install scikit-rare


Documentation for FIBERS Class:
--------------------------------

Documentation for the FIBERS class can be found `here <skrare.html#module-skrare.fibers>`_.

Contact
-------------------------------

Please email sdasariraju23@lawrenceville.org and Ryan.Urbanowicz@cshs.org for any
inquiries related to RARE.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Table of Contents:


   self
   modules

