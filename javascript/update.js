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
