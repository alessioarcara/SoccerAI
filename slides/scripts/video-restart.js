Reveal.addEventListener('ready', () => {
    Reveal.addEventListener('slidechanged', event => {
        const vid = event.currentSlide.querySelector('video');
        if (vid) {
            vid.pause();
            vid.currentTime = 0;
            vid.play();
        }
    });
});
