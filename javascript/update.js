function slider_to_text() {
    let res = Array.from(arguments);
    const ISXLBLOCK = [
        /* base, in01, in02,... */
        true, true, true, true, true, true, true, true, true, true, false, false, false,
        /* mid, out01, out02,... */
        true, true, true, true, true, true, true, true, true, true, false, false, false,
    ];
    let isxl = res[0];

    let selected = [];
    let slider = res.slice(1);
    if (isxl) {
        selected = []
        for (let i = 0; i < slider.length; i++) {
            if (ISXLBLOCK[i]) {
                selected.push(slider[i]);
            }
        }
    } else {
        selected = slider;
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
    setupMergeBlockWeights();
    setupAdjust();

    for (var tab of gradioApp().querySelectorAll('.mm_model_tab')) {
        setupModelTab(tab);
    }
});

})();
