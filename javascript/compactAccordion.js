(function(){

function setupCompactAccordion(accordion) {
    var labelWrap = accordion.querySelector('.label-wrap');

    var isOpen = function() {
        return labelWrap.classList.contains('open');
    };

    var observerAccordionOpen = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutationRecord) {
            accordion.classList.toggle('compact-accordion-open', isOpen());
        });
    });
    observerAccordionOpen.observe(labelWrap, {attributes: true, attributeFilter: ['class']});
}

onUiLoaded(() => {
    for (var accordion of gradioApp().querySelectorAll('.mm-compact-accordion')) {
        setupCompactAccordion(accordion);
    }
});

})();
