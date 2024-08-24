function mm_slider_to_text() {
    let res = Array.from(arguments);

    function is_block_sliders(sdversion) {
        let sliders = Array(1 + 38 + 1 + 38).fill(false); //  base + in 38 blocks + middle + out 38 blocks
        if (sdversion == "v1" || sdversion == "v2") {
            sliders[0] = true; // base
            sliders.splice(1, 12, ...Array(12).fill(true)); // input blocks 00-11
            sliders[39] = true; // middle
            sliders.splice(40, 12, ...Array(12).fill(true)); // output blocks 00-11
        } else if (sdversion == "XL") {
            sliders[0] = true; // base
            sliders.splice(1, 9, ...Array(9).fill(true)); // input blocks 00-08
            sliders[39] = true; // middle
            sliders.splice(40, 9, ...Array(9).fill(true)); // output blocks 00-08
        } else if (sdversion == "v3") {
            sliders[0] = true; // base
            sliders.splice(1, 12, ...Array(12).fill(true)); // joint blocks 00-11
            sliders[39] = false; // no middle
            sliders.splice(40, 12, ...Array(12).fill(true)); // joint blocks 12-23
        } else if (sdversion == "FLUX") {
            sliders[0] = true; // base
            sliders.splice(1, 19, ...Array(19).fill(true)); // double blocks 00-11
            sliders[39] = false; // no middle
            sliders.splice(40, 38, ...Array(38).fill(true)); // single blocks 00-37
        }
        return sliders;
    }

    let sdv = res[0];

    let selected = [];
    const slider = res.slice(1);
    const block_sliders = is_block_sliders(sdv);
    for (let i = 0; i < slider.length; i++) {
        if (block_sliders[i]) {
            selected.push(slider[i]);
        }
    }

    let mbw = null;
    let mbws = gradioApp().querySelectorAll(".mm_mbw textarea");
    for (let i = 0; i < mbws.length; i++) {
        if (mbws[i].parentElement.offsetParent) {
            mbw = mbws[i]
            mbw.value = selected.join(",");
            break;
        }
    }

    /* need to click the set button to work with gradio */
    let btn = null;
    let mbw_set_buttons = gradioApp().querySelectorAll("button.mm_mbw_set");
    for (let i = 0; i < mbw_set_buttons.length; i++) {
        if (mbw_set_buttons[i].parentElement.offsetParent) {
            btn = mbw_set_buttons[i]
            btn.click();
            break;
        }
    }

    return res;
}

function mm_sync_checkpoint() {
    let res = Array.from(arguments);

    if (res[0] && res[1]) {
        let model = res[1].replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#x27;');
        selectCheckpoint(model);
    }
}

var mm_text_to_slider = function() {};
var slider_to_text = mm_slider_to_text;

(function(){

function update_mbw() {
    let mbw = null;
    let mbws = gradioApp().querySelectorAll(".mm_mbw textarea");
    for (let i = 0; i < mbws.length; i++) {
        if (mbws[i].parentElement.offsetParent) {
            mbw = mbws[i]
            break;
        }
    }
    if (mbw == null)
        return;

    /* click the read button of the selected model */
    let btn = null;
    let mbw_read_buttons = gradioApp().querySelectorAll("button.mm_mbw_read");
    for (let i = 0; i < mbw_read_buttons.length; i++) {
        if (mbw_read_buttons[i].parentElement.offsetParent) {
            btn = mbw_read_buttons[i]
            btn.click();
            break;
        }
    }
}

// setup model tab to update merge block weights
function setupModelTab(tab){
    var observer = new MutationObserver(function(mutations) {
        for (var mutation of mutations) {
            if (mutation.target.style.display === 'block') {
                update_mbw();
            }
        }
    });

    observer.observe(tab, {attributes: true, attributeFilter: ['style']});
}

// setup mergeblock weights accordion to read the merge block weights of the current selected model
function setupMergeBlockWeights() {
    let controls = gradioApp().querySelectorAll(".model_mixer_mbws_control");

    var observer = new MutationObserver(function(mutations) {
        for (var mutation of mutations) {
            if (mutation.target.classList.contains('open')) {
                update_mbw();
            }
        }
    });
    for (var control of controls) {
        var labelwrap = control.querySelector('.label-wrap');
        observer.observe(labelwrap, {attributes: true, attributeFilter: ['class']});
    }
}

// click adjust read button
function update_adjust() {
    let adjust = null;
    let adjusts = gradioApp().querySelectorAll(".model_mixer_adjust_control");
    for (let i = 0; i < adjusts.length; i++) {
        if (adjusts[i].parentElement.offsetParent) {
            adjust = adjusts[i]
            break;
        }
    }

    if (adjust) {
        adjust.querySelector("button.mm_adjust_read").click();
    }

}

// setup adjust settings accordion to read the current adjust settings
function setupAdjust() {
    let controls = gradioApp().querySelectorAll(".model_mixer_adjust_control");

    var observer = new MutationObserver(function(mutations) {
        for (var mutation of mutations) {
            if (mutation.target.classList.contains('open')) {
                // update adjust
                update_adjust();
            }
        }
    });
    for (var control of controls) {
        var labelwrap = control.querySelector('.label-wrap');
        observer.observe(labelwrap, {attributes: true, attributeFilter: ['class']});
    }
}

onUiLoaded(function(){
    mm_text_to_slider = update_mbw;

    setupMergeBlockWeights();
    setupAdjust();

    for (var tab of gradioApp().querySelectorAll('.mm_model_tab')) {
        setupModelTab(tab);
    }
});

})();
