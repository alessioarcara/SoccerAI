# Generating Slides

The [slides]() used to present this project were generated using [quarto](https://quarto.org).

To be able to generate those yourself from quarto's source files (those with extension `.qmd`) make sure you have `quarto` installed on your machine with `quarto --version`.
If not, [install quarto cli](https://quarto.org/docs/get-started/).

At this point, you can simply run
```shell
quarto render soccerai.qmd
```
from the `slides` directory, which will output the slide deck (in `html` form) with a root file `index.html` along with auxiliary files located into a directory `soccerai`. 

You can now view the deck in your browser by opening `index.html`.