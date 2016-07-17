# Analysis code for Jiang et al. 2016

This repository contains the code for the simulation of the effect of slice cutting on the measured connection probability matrix as published by Jiang et al. 2015.

To run this code you need python (with numpy, scipy, and pandas) and [datajoint](http://github.com/datajoint/datajoint-python).

We provide a Docker image with all dependencies installed under `atlab/jiang2016`. To run the container you need the database credentials that can be requested via email.
We will then send you a file called `dj_local_conf.json`. With that file you can run the container via

 ```
 docker run -p 8888:8888 -v $(PWD)/dj_local_conf.json:/jiang2016/dj_local_conf.json atlab/jiang2016
 ```

After that a jupyter notebook server should be available from your browser under `localhost:8888`.

# Licenses

<div>

The datasets provided by this repository and through the database linked with the accompanying Docker container are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/3.0/">Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License</a>. This license requires that you contact us before you use the data in your own research. In particular, this means that you have to ask for permission if you intend to publish a new analysis performed with this data (no derivative works-clause).
<br/>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/3.0/"><img alt="Creative Commons License" style="border-width:0" src="http://i.creativecommons.org/l/by-nc-nd/3.0/88x31.png" /></a><br />
</div><br/>

The code in this repository is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a>.
<br/><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="http://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png" /></a><br />
