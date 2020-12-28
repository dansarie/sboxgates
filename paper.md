---
title: 'sboxgates: A program for finding low gate count implementations of S-boxes'
tags:
  - S-box
  - cryptography
authors:
  - name: Marcus Dansarie
    orcid: 0000-0001-9246-0263
    affiliation: "1, 2"
affiliations:
 - name: Swedish Defence University
   index: 1
 - name: University of Skövde, Sweden
   index: 2
date: 28 December 2020
bibliography: paper.bib
---

# Summary

S-boxes are often the only nonlinear components in modern block ciphers. They are commonly selected to comply with very specific criteria in order to make a cipher secure against, for example, linear and differential attacks. An $N \times M$ S-box can be thought of as a lookup table that relates an $M$-bit input value to an $N$-bit output value, or as a set of $N$ boolean functions of $M$ variables [@schneier1996]. <!-- p. 349 -->

Although cipher specifications generally describe S-boxes using their lookup tables, they can also be described as boolean functions or logic gate circuits. `sboxgates`, which is presented here, finds equivalent logic gate circuits for S-boxes, given their lookup table specification. Generated circuits are output in a human-readable XML format. The software can convert the output files into C or CUDA (a programming language for Nvidia GPUs) source code. The generated circuits can also be converted to the DOT graph description language for visualization with Graphviz [@graphviz].

# Statement of need

Knowledge of a low gate count logic gate representation of an S-box can be of interest both when assessing the security of a cipher through cryptanalysis and when implementing it in hardware or software. When the design specification for an S-box is known, analytic approaches can sometimes be used to construct such a representation. The most notable case of this is the AES cipher where a very efficient representation of the S-box has been constructed in this manner [@canright2005]. However, this is not possible in many cases, such as when the design specification is unknown or if the S-box is a randomly generated permutation.

While finding a large, inefficient, logic circuit representation is trivial, finding the representation with the fewest possible gates is an NP-complete problem [@knuth2015tacp4f6]. The best known way to find a low gate count logic circuit representation for an S-box given its lookup table is to use Kwan's algorithm, which performs a heuristic search. Although not optimal, it has been shown to produce significantly better results than previous approaches [@kwan2000].

`sboxgates` implements Kwan's algorithm and supports generation of logic circuits for S-boxes with up to 8 input bits using any subset of the 16 possible two-input boolean functions. Additionally, the program can generate circuits that include three-bit lookup tables (LUTs). The LUT search function is parallelized using MPI [@walker1996].

The generated logic circuit representation of an S-box can be directly used in applications such as: creating bitslice implementations in software for CPUs and GPUs, creating small chip area or low gate count S-boxes for application specific integrated circuits (ASICs) or field programmable gate arrays (FPGAs), and compact satisfiability (SAT) problem generation.

A bitslice software implementation was first described by Biham [-@biham1994]. It implements a cipher in a way that mimics a parallel hardware implementation. The number of parallel operations is equivalent to the platform register size, which can be up to 512 bits on modern machines. Some operations, such as bit permutations, have effectively no cost in bitslice implementations. For these reasons, bitslice implementations are generally many times faster than conventional implementations. The primary contributor to their time complexity is the size and efficiency of the S-box logic circuit [@biham1994]. Modern Nvidia GPUs have an instruction (LOP3.LUT) that evaluates three-bit lookup tables. This makes them a very attractive target for bitslice cipher implementations that use LUTs.

In addition to bitslice implementations in software, which attempt to mimic hardware implementations, designs of actual hardware such as application specific integrated circuits (ASICs) or field programmable gate arrays (FPGAs) can also be made more efficient by small equivalent logic gate circuits for S-boxes.

In algebraic cryptanalysis, one attack method is to model a cipher along with its inputs and outputs as a SAT problem. This can be used to find, for example, weak keys in block ciphers or preimates in hash functions [@lafitte2014]. SAT problems are typically expressed in conjunctive normal form (CNF) and logic circuits can quickly be converted into CNF using the Tseytin transform [@knuth2015tacp4f6]. Thus, an efficient logic gate representation of an S-box can be transformed into an efficient CNF representation. CNF representations can in turn be transformed into a system of equations in GF(2) [@lafitte2014].

The only known software with similar functionality to `sboxgates` is `SBOXDiscovery` which is restricted to generating logic circuit representations of the DES S-boxes [@sboxdiscovery]. The software has been abandoned by its original author. Many of the optimizations of Kwan's algorithm made in `SBOXDiscovery` have been included in `sboxgates`.

# References
