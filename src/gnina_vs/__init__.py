"""
gnina_vs
===========
A modular virtual-screening pipeline: protein preparation (with RCSB
auto-fetch), ligand preparation, three-tier active-site detection, parallel
Gnina docking, and results reporting/export.

See ``gnina_vs.cli`` for the command-line entry point, or import the
individual step modules (``protein``, ``ligand``, ``pocket``, ``docking``,
``reporter``) directly for programmatic use.
"""

__version__ = "1.0.0"
