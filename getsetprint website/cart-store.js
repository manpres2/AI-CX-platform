/*
 * Shared cart storage used across index.html, product.html and cart.html.
 * Uses localStorage so the cart persists as a visitor moves between pages.
 * This is a front-end demo only — there is no server, no payment gateway,
 * and no order is actually placed. Wire "Proceed to Checkout" up to a real
 * payment provider (Razorpay / Stripe / etc.) before taking real orders.
 */
(function (window) {
  var STORAGE_KEY = 'getsetprint_cart';

  function fmt(n) {
    return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function readCart() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }

  function writeCart(items) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    } catch (e) {
      /* storage unavailable — cart just won't persist */
    }
  }

  function addToCart(item) {
    var items = readCart();
    var existing = items.find(function (it) {
      return it.id === item.id && it.variant === item.variant;
    });
    if (existing) {
      existing.qty += item.qty;
    } else {
      items.push(item);
    }
    writeCart(items);
    updateCartBadge();
  }

  function itemCount() {
    return readCart().reduce(function (sum, it) { return sum + it.qty; }, 0);
  }

  function updateCartBadge() {
    var count = itemCount();
    document.querySelectorAll('[data-cart-badge]').forEach(function (el) {
      el.textContent = count;
      el.style.display = count > 0 ? 'flex' : 'none';
    });
  }

  window.GetSetPrintCart = {
    fmt: fmt,
    read: readCart,
    write: writeCart,
    add: addToCart,
    count: itemCount,
    updateBadge: updateCartBadge,
  };

  document.addEventListener('DOMContentLoaded', updateCartBadge);
})(window);
