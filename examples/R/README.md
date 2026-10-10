# R loader

`load_solete.R` provides `load_solete(path)`, a minimal R function that reads
one of SOLETE's real HDF5 files (`examples/SOLETE_short.h5`, `SOLETE_Pombo_60min.h5`,
or any sibling `SOLETE_Pombo_<resolution>.h5` produced the same way) into a
plain R `data.frame`, using the [`hdf5r`](https://cran.r-project.org/package=hdf5r)
package.

That was generated with an LLM as I've never worked with R. Let me know if there are any problems.
