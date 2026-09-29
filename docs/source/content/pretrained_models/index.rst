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


.. _pretrained_models_home:

Released OpenSpliceAI models
====================================

This page provides all released models trained with different flanking sequence lengths using OpenSpliceAI. We trained OpenSpliceAI with four flanking sequence configurations — 80 nt, 400 nt, 2000 nt, and 10000 nt — and strongly recommend using the 10000 nt model for optimal performance. The other configurations are available for experimental and research purposes. Browse the models below to choose the one that best meets your needs.

.. admonition:: LiftOn examples
    :class: note

    * :ref:`human_mane_spliceai`
    * :ref:`mouse_spliceai`
    * :ref:`zebrafish_spliceai`
    * :ref:`bee_insect_spliceai`
    * :ref:`thale_cress_plant_spliceai`
    
    .. * :ref:`human_refseq_spliceai`


.. toctree::
    :hidden:

    GRCh38_MANE
    mouse
    zebrafish
    bee_insect
    arabidopsis_plant


|
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