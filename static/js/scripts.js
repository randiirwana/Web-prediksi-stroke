// Aktifkan scrollspy Bootstrap
const mainNav = document.getElementById('mainNav');
if (mainNav) {
    new bootstrap.ScrollSpy(document.body, {
        target: '#mainNav',
        offset: 80,
    });
}

// Tutup nav collapse saat link diklik (untuk mobile)
const navbarToggler = document.querySelector('.navbar-toggler');
const responsiveNavItems = document.querySelectorAll('#navbarResponsive .nav-link');
responsiveNavItems.forEach((item) => {
    item.addEventListener('click', () => {
        if (window.getComputedStyle(navbarToggler).display !== 'none') {
            navbarToggler.click();
        }
    });
});

// Efek shrink nav ketika scroll
const shrinkNav = () => {
    if (!mainNav) return;
    if (window.scrollY > 80) {
        mainNav.classList.add('navbar-shrink');
    } else {
        mainNav.classList.remove('navbar-shrink');
    }
};

shrinkNav();
document.addEventListener('scroll', shrinkNav);

