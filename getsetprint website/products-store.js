/*
 * Loads product data from products.json (shared by index.html and product.html).
 * To add a product: add one entry to products.json. To add its photo: drop an
 * image file in images/ named to match that entry's "image" path — until the
 * file exists, a placeholder icon is shown automatically.
 */
(function (window) {
  var cache = null;

  function loadProducts() {
    if (!cache) {
      cache = fetch('products.json').then(function (r) { return r.json(); });
    }
    return cache;
  }

  function getById(id) {
    return loadProducts().then(function (products) {
      return products.find(function (p) { return p.id === id; }) || null;
    });
  }

  function thumbHtml(p, size) {
    size = size || 76;
    return (
      '<img src="' + p.image + '" alt="' + p.name + '" loading="lazy" ' +
        'style="width:100%;height:100%;object-fit:cover;" ' +
        'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">' +
      '<div class="thumb-fallback" style="display:none;align-items:center;justify-content:center;width:100%;height:100%;">' +
        '<span style="display:inline-block;width:' + size + 'px;height:' + size + 'px;">' + p.icon + '</span>' +
      '</div>'
    );
  }

  window.GetSetPrintProducts = {
    all: loadProducts,
    getById: getById,
    thumbHtml: thumbHtml,
  };
})(window);
