function mm_vxa_prepare() {
    var res = Array.from(arguments);

    // get the current selected gallery id
    var idx = selected_gallery_index();
    res[1] = idx; // gallery id

    return res;
}

