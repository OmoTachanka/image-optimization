const main_container = document.querySelector('.main-container');
document.querySelector('.slider').addEventListener('input', (e) => {
    main_container.style.setProperty('--position', `${e.target.value}%`);
})
