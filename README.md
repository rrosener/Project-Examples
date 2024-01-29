# Public Project Examples
An example of code for various public projects.

## Senior Thesis 
I am currently working on a senior thesis for honors achievment in the astrophysics B.S. program at UChicago. My advisor is [Michael Zhang](https://astro.uchicago.edu/~mz/) in Jacob Bean's Exoplanet Group.

## Classwork Code
I have a selection of code written for various homework assignments, particularly for ASTR258 Exoplanets.

# Private Project Examples
Unfortunately, much of my project work is private and cannot be shared, but I will share some relevant information about those projects below.

## QACITS Revamp
NIRC2 is an infrared imaging instrument on the Keck II telescope at W.M. Keck Observatory, in Hawai'i. In the summer of 2023, I was an intern at Keck working to transfer the QACITS code for NIRC2 from IDL into Python 3, creating a robust backend while also implementing much more comprehensive debugging and logging capabilities. I mainly worked with staff astronomer Carlos Alvarez on development and testing.

QACITS is an exoplanet imaging code interfacing with the NIRC2 coronagraphy mode, a principal part of my work was implementing [the QACITS algorithm](https://www.aanda.org/articles/aa/full_html/2015/12/aa27102-15/aa27102-15.html) to radpily adjust opticial equipment positioning in order to optimize observations. I engaged in extensive research on documentation, development, and computer science infrastructure best practices, in order to tailor my code to perfectly suit the needs of the astronomers who would be using and maintaining it. The QACITS software--a proven tool aiding in novel exoplanet science--aims to acquire, process, and analyze astronomical data while on-sky observing, controlling the telescope, cameras, and adaptive optics systems.

As a result of my work (and copious live-testing on equipment), the QACITS backend was successfully created and tested. Some additional information on QACITS can be found in the [NIRC2 Observer's Manual.](https://www2.keck.hawaii.edu/inst/nirc2/ObserversManual.html) There is not any public information I can link to at the moment, but QACITS should formally roll out soon, and there will be a new QACITS User's Manual I am working on as well.

## COOL-LAMPS Quasar Search
I am a member of the [COOL-LAMPS research collaboration](https://coollamps.github.io/index.html), which engages in astronomy research into strong lensing using a combination of archival data and new observations taken with telescopes like Hubble. 

My main focus was on the search for widely-separated lensed quasars (WSLQs), an extremely rare class of astronomical object which the group has discovered 5 of the 9 published examples of (for example, see [Napier et al 2023](https://inspirehep.net/literature/2662079)). The most recent of these was discovered while I was working on optimizing the candidate search algorithm, which will be published in the forthcoming paper COOL-LAMPS VI.

I worked in IDL on the pipeline processing extremely large amounts of archival, astronomical data across a huge parameter space, in order to find WSLQ candidates, of which we have very few samples. On multiple observing runs, I helped narrow the candidate list from millions of objects to a shortlist of 4-5, incorporating knowledge about known properties of WLSQs, statistical properties of astronomical observations, experimentation with differently-correlated parameters, and a bit of visual searching. 
