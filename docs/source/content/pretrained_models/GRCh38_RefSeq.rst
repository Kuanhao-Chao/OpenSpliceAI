:orphan:

.. raw:: html

    <script type="text/javascript">
        let mutation_lvl_1_fuc = function(mutations) {
            var dark = document.body.dataset.theme == 'dark';

            if (document.body.dataset.theme == 'auto') {
                dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            }
            
            document.getElementsByClassName('sidebar_ccb')[0].src = dark ? '../../_static/JHU_ccb-white.png' : "../../_static/JHU_ccb-dark.png";
            document.getElementsByClassName('sidebar_wse')[0].src = dark ? '../../_static/JHU_wse-white.png' : "../../_static/JHU_wse-dark.png";



            for (let i=0; i < document.getElementsByClassName('summary-title').length; i++) {
                console.log(">> document.getElementsByClassName('summary-title')[i]: ", document.getElementsByClassName('summary-title')[i]);

                if (dark) {
                    document.getElementsByClassName('summary-title')[i].classList = "summary-title card-header bg-dark font-weight-bolder";
                    document.getElementsByClassName('summary-content')[i].classList = "summary-content card-body bg-dark text-left docutils";
                } else {
                    document.getElementsByClassName('summary-title')[i].classList = "summary-title card-header bg-light font-weight-bolder";
                    document.getElementsByClassName('summary-content')[i].classList = "summary-content card-body bg-light text-left docutils";
                }
            }

        }
        document.addEventListener("DOMContentLoaded", mutation_lvl_1_fuc);
        var observer = new MutationObserver(mutation_lvl_1_fuc)
        observer.observe(document.body, {attributes: true, attributeFilter: ['data-theme']});
        console.log(document.body);
    </script>


|


.. |OSAI_RefSeq_10000| raw:: html

   <b>OSAI<sub>RefSeq</sub>-10000nt</b>


.. _human_refseq_spliceai:

Human (RefSeq)
===================================================================


Released models
+++++++++++++++++++++++++++++++++++

We trained OpenSpliceAI using four different flanking sequence lengths: 80 nt, 400 nt, 2000 nt, and 10000 nt. We strongly recommend using |OSAI_RefSeq_10000| for best performance. The other models are suitable for experimental / research purposes.

.. raw:: html

    <ul>
    <li><b>OSAI<sub>RefSeq</sub>-10000nt</b>
        <ul>
        <li>
            <a href="ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-RefSeq/SpliceAI-RefSeq-10000nt.pt" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" x="0px" y="0px"
                viewBox="0 0 100 100" width="15" height="15" class="icon outbound">
                <path fill="currentColor" d="M18.8,85.1h56l0,0c2.2,0,4-1.8,4-4v-32h-8v28h-48v-48h28v-8h-32l0,0
                c-2.2,0-4,1.8-4,4v56C14.8,83.3,16.6,85.1,18.8,85.1z"></path>
                <polygon fill="currentColor" points="45.7,48.7 51.3,54.3 77.2,28.5 77.2,37.2 85.2,37.2 85.2,14.9 62.8,14.9 62.8,22.9 71.5,22.9"></polygon>
            </svg>
            Model 1
            </a>
        </li>
        </ul>
    </li>
    <li><b>OSAI<sub>RefSeq</sub>-2000nt</b>
        <ul>
        <li>
            <a href="ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-RefSeq/SpliceAI-RefSeq-2000nt.pt" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" x="0px" y="0px"
                viewBox="0 0 100 100" width="15" height="15" class="icon outbound">
                <path fill="currentColor" d="M18.8,85.1h56l0,0c2.2,0,4-1.8,4-4v-32h-8v28h-48v-48h28v-8h-32l0,0
                c-2.2,0-4,1.8-4,4v56C14.8,83.3,16.6,85.1,18.8,85.1z"></path>
                <polygon fill="currentColor" points="45.7,48.7 51.3,54.3 77.2,28.5 77.2,37.2 85.2,37.2 85.2,14.9 62.8,14.9 62.8,22.9 71.5,22.9"></polygon>
            </svg>
            Model 1
            </a>
        </li>
        </ul>
    </li>
    <li><b>OSAI<sub>RefSeq</sub>-400nt</b>
        <ul>
        <li>
            <a href="ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-RefSeq/SpliceAI-RefSeq-400nt.pt" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" x="0px" y="0px"
                viewBox="0 0 100 100" width="15" height="15" class="icon outbound">
                <path fill="currentColor" d="M18.8,85.1h56l0,0c2.2,0,4-1.8,4-4v-32h-8v28h-48v-48h28v-8h-32l0,0
                c-2.2,0-4,1.8-4,4v56C14.8,83.3,16.6,85.1,18.8,85.1z"></path>
                <polygon fill="currentColor" points="45.7,48.7 51.3,54.3 77.2,28.5 77.2,37.2 85.2,37.2 85.2,14.9 62.8,14.9 62.8,22.9 71.5,22.9"></polygon>
            </svg>
            Model 1
            </a>
        </li>
        </ul>
    </li>
    <li><b>OSAI<sub>RefSeq</sub>-80nt</b>
        <ul>
        <li>
            <a href="ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-RefSeq/SpliceAI-RefSeq-80nt.pt" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" x="0px" y="0px"
                viewBox="0 0 100 100" width="15" height="15" class="icon outbound">
                <path fill="currentColor" d="M18.8,85.1h56l0,0c2.2,0,4-1.8,4-4v-32h-8v28h-48v-48h28v-8h-32l0,0
                c-2.2,0-4,1.8-4,4v56C14.8,83.3,16.6,85.1,18.8,85.1z"></path>
                <polygon fill="currentColor" points="45.7,48.7 51.3,54.3 77.2,28.5 77.2,37.2 85.2,37.2 85.2,14.9 62.8,14.9 62.8,22.9 71.5,22.9"></polygon>
            </svg>
            Model 1
            </a>
        </li>
        </ul>
    </li>
    </ul>



|
|

.. _alignment-whats-next:

.. What's next?
.. +++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. Congratulations! You have finished this tutorial.

.. seealso::
    
    * :ref:`behind-the-scenes-splam` to understand how LiftOn is designed
    * :ref:`Q&A` to check out some common questions


|
|
|
|

.. image:: ../../_images/jhu-logo-dark.png
   :alt: My Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ../../_images/jhu-logo-white.png
   :alt: My Logo
   :class: logo, header-image only-dark
   :align: center