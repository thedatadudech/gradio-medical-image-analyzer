const {
  SvelteComponent: U,
  append_hydration: m,
  attr: f,
  children: y,
  claim_element: p,
  claim_space: k,
  claim_text: b,
  detach: c,
  element: h,
  get_svelte_dataset: R,
  init: j,
  insert_hydration: d,
  noop: I,
  safe_not_equal: z,
  set_data: E,
  space: C,
  src_url_equal: w,
  text: g,
  toggle_class: v
} = window.__gradio__svelte__internal;
function B(n) {
  let e, a = "No example";
  return {
    c() {
      e = h("div"), e.textContent = a, this.h();
    },
    l(l) {
      e = p(l, "DIV", { class: !0, "data-svelte-h": !0 }), R(e) !== "svelte-1xsigcs" && (e.textContent = a), this.h();
    },
    h() {
      f(e, "class", "empty-example svelte-16yp9bf");
    },
    m(l, t) {
      d(l, e, t);
    },
    p: I,
    d(l) {
      l && c(e);
    }
  };
}
function F(n) {
  let e, a, l = (
    /*value*/
    n[0].image && S(n)
  ), t = (
    /*value*/
    n[0].analysis && A(n)
  );
  return {
    c() {
      e = h("div"), l && l.c(), a = C(), t && t.c(), this.h();
    },
    l(i) {
      e = p(i, "DIV", { class: !0 });
      var s = y(e);
      l && l.l(s), a = k(s), t && t.l(s), s.forEach(c), this.h();
    },
    h() {
      f(e, "class", "example-content svelte-16yp9bf");
    },
    m(i, s) {
      d(i, e, s), l && l.m(e, null), m(e, a), t && t.m(e, null);
    },
    p(i, s) {
      /*value*/
      i[0].image ? l ? l.p(i, s) : (l = S(i), l.c(), l.m(e, a)) : l && (l.d(1), l = null), /*value*/
      i[0].analysis ? t ? t.p(i, s) : (t = A(i), t.c(), t.m(e, null)) : t && (t.d(1), t = null);
    },
    d(i) {
      i && c(e), l && l.d(), t && t.d();
    }
  };
}
function S(n) {
  let e;
  function a(i, s) {
    return typeof /*value*/
    i[0].image == "string" ? K : (
      /*value*/
      i[0].image.url ? J : H
    );
  }
  let l = a(n), t = l(n);
  return {
    c() {
      e = h("div"), t.c(), this.h();
    },
    l(i) {
      e = p(i, "DIV", { class: !0 });
      var s = y(e);
      t.l(s), s.forEach(c), this.h();
    },
    h() {
      f(e, "class", "image-preview svelte-16yp9bf");
    },
    m(i, s) {
      d(i, e, s), t.m(e, null);
    },
    p(i, s) {
      l === (l = a(i)) && t ? t.p(i, s) : (t.d(1), t = l(i), t && (t.c(), t.m(e, null)));
    },
    d(i) {
      i && c(e), t.d();
    }
  };
}
function H(n) {
  let e, a = "📷 Image";
  return {
    c() {
      e = h("div"), e.textContent = a, this.h();
    },
    l(l) {
      e = p(l, "DIV", { class: !0, "data-svelte-h": !0 }), R(e) !== "svelte-1hvroc5" && (e.textContent = a), this.h();
    },
    h() {
      f(e, "class", "placeholder svelte-16yp9bf");
    },
    m(l, t) {
      d(l, e, t);
    },
    p: I,
    d(l) {
      l && c(e);
    }
  };
}
function J(n) {
  let e, a;
  return {
    c() {
      e = h("img"), this.h();
    },
    l(l) {
      e = p(l, "IMG", { src: !0, alt: !0, class: !0 }), this.h();
    },
    h() {
      w(e.src, a = /*value*/
      n[0].image.url) || f(e, "src", a), f(e, "alt", "Medical scan example"), f(e, "class", "svelte-16yp9bf");
    },
    m(l, t) {
      d(l, e, t);
    },
    p(l, t) {
      t & /*value*/
      1 && !w(e.src, a = /*value*/
      l[0].image.url) && f(e, "src", a);
    },
    d(l) {
      l && c(e);
    }
  };
}
function K(n) {
  let e, a;
  return {
    c() {
      e = h("img"), this.h();
    },
    l(l) {
      e = p(l, "IMG", { src: !0, alt: !0, class: !0 }), this.h();
    },
    h() {
      w(e.src, a = /*value*/
      n[0].image) || f(e, "src", a), f(e, "alt", "Medical scan example"), f(e, "class", "svelte-16yp9bf");
    },
    m(l, t) {
      d(l, e, t);
    },
    p(l, t) {
      t & /*value*/
      1 && !w(e.src, a = /*value*/
      l[0].image) && f(e, "src", a);
    },
    d(l) {
      l && c(e);
    }
  };
}
function A(n) {
  var r, _, D;
  let e, a, l, t = (
    /*value*/
    n[0].analysis.modality && P(n)
  ), i = (
    /*value*/
    ((r = n[0].analysis.point_analysis) == null ? void 0 : r.tissue_type) && q(n)
  ), s = (
    /*value*/
    ((D = (_ = n[0].analysis.segmentation) == null ? void 0 : _.interpretation) == null ? void 0 : D.obesity_risk) && G(n)
  );
  return {
    c() {
      e = h("div"), t && t.c(), a = C(), i && i.c(), l = C(), s && s.c(), this.h();
    },
    l(o) {
      e = p(o, "DIV", { class: !0 });
      var u = y(e);
      t && t.l(u), a = k(u), i && i.l(u), l = k(u), s && s.l(u), u.forEach(c), this.h();
    },
    h() {
      f(e, "class", "analysis-preview svelte-16yp9bf");
    },
    m(o, u) {
      d(o, e, u), t && t.m(e, null), m(e, a), i && i.m(e, null), m(e, l), s && s.m(e, null);
    },
    p(o, u) {
      var V, M, N;
      /*value*/
      o[0].analysis.modality ? t ? t.p(o, u) : (t = P(o), t.c(), t.m(e, a)) : t && (t.d(1), t = null), /*value*/
      (V = o[0].analysis.point_analysis) != null && V.tissue_type ? i ? i.p(o, u) : (i = q(o), i.c(), i.m(e, l)) : i && (i.d(1), i = null), /*value*/
      (N = (M = o[0].analysis.segmentation) == null ? void 0 : M.interpretation) != null && N.obesity_risk ? s ? s.p(o, u) : (s = G(o), s.c(), s.m(e, null)) : s && (s.d(1), s = null);
    },
    d(o) {
      o && c(e), t && t.d(), i && i.d(), s && s.d();
    }
  };
}
function P(n) {
  let e, a = (
    /*value*/
    n[0].analysis.modality + ""
  ), l;
  return {
    c() {
      e = h("span"), l = g(a), this.h();
    },
    l(t) {
      e = p(t, "SPAN", { class: !0 });
      var i = y(e);
      l = b(i, a), i.forEach(c), this.h();
    },
    h() {
      f(e, "class", "modality-badge svelte-16yp9bf");
    },
    m(t, i) {
      d(t, e, i), m(e, l);
    },
    p(t, i) {
      i & /*value*/
      1 && a !== (a = /*value*/
      t[0].analysis.modality + "") && E(l, a);
    },
    d(t) {
      t && c(e);
    }
  };
}
function q(n) {
  let e, a = (
    /*value*/
    (n[0].analysis.point_analysis.tissue_type.icon || "") + ""
  ), l, t, i = (
    /*value*/
    (n[0].analysis.point_analysis.tissue_type.type || "Unknown") + ""
  ), s;
  return {
    c() {
      e = h("span"), l = g(a), t = C(), s = g(i), this.h();
    },
    l(r) {
      e = p(r, "SPAN", { class: !0 });
      var _ = y(e);
      l = b(_, a), t = k(_), s = b(_, i), _.forEach(c), this.h();
    },
    h() {
      f(e, "class", "tissue-type svelte-16yp9bf");
    },
    m(r, _) {
      d(r, e, _), m(e, l), m(e, t), m(e, s);
    },
    p(r, _) {
      _ & /*value*/
      1 && a !== (a = /*value*/
      (r[0].analysis.point_analysis.tissue_type.icon || "") + "") && E(l, a), _ & /*value*/
      1 && i !== (i = /*value*/
      (r[0].analysis.point_analysis.tissue_type.type || "Unknown") + "") && E(s, i);
    },
    d(r) {
      r && c(e);
    }
  };
}
function G(n) {
  let e, a, l = (
    /*value*/
    n[0].analysis.segmentation.interpretation.obesity_risk + ""
  ), t, i;
  return {
    c() {
      e = h("span"), a = g("Risk: "), t = g(l), this.h();
    },
    l(s) {
      e = p(s, "SPAN", { class: !0 });
      var r = y(e);
      a = b(r, "Risk: "), t = b(r, l), r.forEach(c), this.h();
    },
    h() {
      f(e, "class", i = "risk-badge risk-" + /*value*/
      n[0].analysis.segmentation.interpretation.obesity_risk + " svelte-16yp9bf");
    },
    m(s, r) {
      d(s, e, r), m(e, a), m(e, t);
    },
    p(s, r) {
      r & /*value*/
      1 && l !== (l = /*value*/
      s[0].analysis.segmentation.interpretation.obesity_risk + "") && E(t, l), r & /*value*/
      1 && i !== (i = "risk-badge risk-" + /*value*/
      s[0].analysis.segmentation.interpretation.obesity_risk + " svelte-16yp9bf") && f(e, "class", i);
    },
    d(s) {
      s && c(e);
    }
  };
}
function L(n) {
  let e;
  function a(i, s) {
    return (
      /*value*/
      i[0] ? F : B
    );
  }
  let l = a(n), t = l(n);
  return {
    c() {
      e = h("div"), t.c(), this.h();
    },
    l(i) {
      e = p(i, "DIV", { class: !0 });
      var s = y(e);
      t.l(s), s.forEach(c), this.h();
    },
    h() {
      f(e, "class", "example-container svelte-16yp9bf"), v(
        e,
        "table",
        /*type*/
        n[1] === "table"
      ), v(
        e,
        "gallery",
        /*type*/
        n[1] === "gallery"
      ), v(
        e,
        "selected",
        /*selected*/
        n[2]
      );
    },
    m(i, s) {
      d(i, e, s), t.m(e, null);
    },
    p(i, [s]) {
      l === (l = a(i)) && t ? t.p(i, s) : (t.d(1), t = l(i), t && (t.c(), t.m(e, null))), s & /*type*/
      2 && v(
        e,
        "table",
        /*type*/
        i[1] === "table"
      ), s & /*type*/
      2 && v(
        e,
        "gallery",
        /*type*/
        i[1] === "gallery"
      ), s & /*selected*/
      4 && v(
        e,
        "selected",
        /*selected*/
        i[2]
      );
    },
    i: I,
    o: I,
    d(i) {
      i && c(e), t.d();
    }
  };
}
function O(n, e, a) {
  let { value: l } = e, { type: t } = e, { selected: i = !1 } = e;
  return n.$$set = (s) => {
    "value" in s && a(0, l = s.value), "type" in s && a(1, t = s.type), "selected" in s && a(2, i = s.selected);
  }, [l, t, i];
}
class Q extends U {
  constructor(e) {
    super(), j(this, e, O, L, z, { value: 0, type: 1, selected: 2 });
  }
}
export {
  Q as default
};
