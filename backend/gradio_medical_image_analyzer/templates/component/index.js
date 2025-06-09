const {
  HtmlTagHydration: Te,
  SvelteComponent: Ee,
  add_render_callback: ye,
  append_hydration: g,
  assign: Ne,
  attr: R,
  check_outros: Ue,
  children: V,
  claim_component: te,
  claim_element: p,
  claim_html_tag: Se,
  claim_space: F,
  claim_text: he,
  create_component: le,
  destroy_component: ne,
  detach: h,
  element: v,
  empty: me,
  get_spread_object: De,
  get_spread_update: Ae,
  get_svelte_dataset: j,
  group_outros: Fe,
  init: Me,
  insert_hydration: U,
  listen: fe,
  mount_component: ie,
  noop: Le,
  run_all: Pe,
  safe_not_equal: je,
  select_option: de,
  select_value: ke,
  set_data: qe,
  set_input_value: Q,
  set_style: q,
  space: M,
  src_url_equal: Ce,
  text: pe,
  transition_in: W,
  transition_out: Y
} = window.__gradio__svelte__internal;
function Ve(o) {
  let e, t, l, a, s, r, u = "CT", m, I = "CR (X-Ray)", C, z = "DX (X-Ray)", k, B = "RX (X-Ray)", c, b = "DR (X-Ray)", S, w, O, Z = "Point Analysis", T, G = "Fat Segmentation (CT)", E, X = "Full Analysis", H, D, A, se, ae, L, oe, x, $, re, _e;
  l = new IconButton({ props: { Icon: UploadIcon } }), l.$on(
    "click",
    /*handle_clear*/
    o[21]
  );
  let n = (
    /*uploaded_file*/
    o[15] && we(o)
  ), i = (
    /*visual_report*/
    o[17] && Oe(o)
  ), f = (
    /*analysis_mode*/
    o[14] === "structured" && /*analysis_results*/
    o[16] && Re(o)
  );
  return {
    c() {
      e = v("div"), t = v("div"), le(l.$$.fragment), a = M(), s = v("select"), r = v("option"), r.textContent = u, m = v("option"), m.textContent = I, C = v("option"), C.textContent = z, k = v("option"), k.textContent = B, c = v("option"), c.textContent = b, S = M(), w = v("select"), O = v("option"), O.textContent = Z, T = v("option"), T.textContent = G, E = v("option"), E.textContent = X, H = M(), D = v("label"), A = v("input"), se = pe(`
					Show ROI`), ae = M(), L = v("div"), n && n.c(), oe = M(), i && i.c(), x = M(), f && f.c(), this.h();
    },
    l(_) {
      e = p(_, "DIV", { class: !0 });
      var d = V(e);
      t = p(d, "DIV", { class: !0 });
      var N = V(t);
      te(l.$$.fragment, N), a = F(N), s = p(N, "SELECT", { class: !0 });
      var P = V(s);
      r = p(P, "OPTION", { "data-svelte-h": !0 }), j(r) !== "svelte-1uwvdsi" && (r.textContent = u), m = p(P, "OPTION", { "data-svelte-h": !0 }), j(m) !== "svelte-iiiy1a" && (m.textContent = I), C = p(P, "OPTION", { "data-svelte-h": !0 }), j(C) !== "svelte-16a5ymm" && (C.textContent = z), k = p(P, "OPTION", { "data-svelte-h": !0 }), j(k) !== "svelte-bjfw5q" && (k.textContent = B), c = p(P, "OPTION", { "data-svelte-h": !0 }), j(c) !== "svelte-121hs3y" && (c.textContent = b), P.forEach(h), S = F(N), w = p(N, "SELECT", { class: !0 });
      var K = V(w);
      O = p(K, "OPTION", { "data-svelte-h": !0 }), j(O) !== "svelte-17yivkd" && (O.textContent = Z), T = p(K, "OPTION", { "data-svelte-h": !0 }), j(T) !== "svelte-cf7bpu" && (T.textContent = G), E = p(K, "OPTION", { "data-svelte-h": !0 }), j(E) !== "svelte-d3m60d" && (E.textContent = X), K.forEach(h), H = F(N), D = p(N, "LABEL", { class: !0 });
      var ee = V(D);
      A = p(ee, "INPUT", { type: !0 }), se = he(ee, `
					Show ROI`), ee.forEach(h), N.forEach(h), ae = F(d), L = p(d, "DIV", { class: !0 });
      var ce = V(L);
      n && n.l(ce), ce.forEach(h), oe = F(d), i && i.l(d), x = F(d), f && f.l(d), d.forEach(h), this.h();
    },
    h() {
      r.__value = "CT", Q(r, r.__value), m.__value = "CR", Q(m, m.__value), C.__value = "DX", Q(C, C.__value), k.__value = "RX", Q(k, k.__value), c.__value = "DR", Q(c, c.__value), R(s, "class", "modality-select svelte-197pbtm"), /*modality*/
      o[1] === void 0 && ye(() => (
        /*select0_change_handler*/
        o[27].call(s)
      )), O.__value = "analyze_point", Q(O, O.__value), T.__value = "segment_fat", Q(T, T.__value), E.__value = "full_analysis", Q(E, E.__value), R(w, "class", "task-select svelte-197pbtm"), /*task*/
      o[2] === void 0 && ye(() => (
        /*select1_change_handler*/
        o[28].call(w)
      )), R(A, "type", "checkbox"), R(D, "class", "roi-toggle svelte-197pbtm"), R(t, "class", "controls svelte-197pbtm"), R(L, "class", "image-container svelte-197pbtm"), R(e, "class", "analyzer-container svelte-197pbtm");
    },
    m(_, d) {
      U(_, e, d), g(e, t), ie(l, t, null), g(t, a), g(t, s), g(s, r), g(s, m), g(s, C), g(s, k), g(s, c), de(
        s,
        /*modality*/
        o[1],
        !0
      ), g(t, S), g(t, w), g(w, O), g(w, T), g(w, E), de(
        w,
        /*task*/
        o[2],
        !0
      ), g(t, H), g(t, D), g(D, A), A.checked = /*show_roi*/
      o[19], g(D, se), g(e, ae), g(e, L), n && n.m(L, null), g(e, oe), i && i.m(e, null), g(e, x), f && f.m(e, null), $ = !0, re || (_e = [
        fe(
          s,
          "change",
          /*select0_change_handler*/
          o[27]
        ),
        fe(
          w,
          "change",
          /*select1_change_handler*/
          o[28]
        ),
        fe(
          A,
          "change",
          /*input_change_handler*/
          o[29]
        ),
        fe(
          L,
          "click",
          /*handle_roi_click*/
          o[22]
        )
      ], re = !0);
    },
    p(_, d) {
      d[0] & /*modality*/
      2 && de(
        s,
        /*modality*/
        _[1]
      ), d[0] & /*task*/
      4 && de(
        w,
        /*task*/
        _[2]
      ), d[0] & /*show_roi*/
      524288 && (A.checked = /*show_roi*/
      _[19]), /*uploaded_file*/
      _[15] ? n ? n.p(_, d) : (n = we(_), n.c(), n.m(L, null)) : n && (n.d(1), n = null), /*visual_report*/
      _[17] ? i ? i.p(_, d) : (i = Oe(_), i.c(), i.m(e, x)) : i && (i.d(1), i = null), /*analysis_mode*/
      _[14] === "structured" && /*analysis_results*/
      _[16] ? f ? f.p(_, d) : (f = Re(_), f.c(), f.m(e, null)) : f && (f.d(1), f = null);
    },
    i(_) {
      $ || (W(l.$$.fragment, _), $ = !0);
    },
    o(_) {
      Y(l.$$.fragment, _), $ = !1;
    },
    d(_) {
      _ && h(e), ne(l), n && n.d(), i && i.d(), f && f.d(), re = !1, Pe(_e);
    }
  };
}
function Xe(o) {
  let e, t;
  return e = new Upload({
    props: {
      filetype: "*",
      root: (
        /*root*/
        o[8]
      ),
      dragging: Qe,
      $$slots: { default: [Be] },
      $$scope: { ctx: o }
    }
  }), e.$on(
    "load",
    /*handle_upload*/
    o[20]
  ), {
    c() {
      le(e.$$.fragment);
    },
    l(l) {
      te(e.$$.fragment, l);
    },
    m(l, a) {
      ie(e, l, a), t = !0;
    },
    p(l, a) {
      const s = {};
      a[0] & /*root*/
      256 && (s.root = /*root*/
      l[8]), a[0] & /*gradio*/
      8192 | a[1] & /*$$scope*/
      4 && (s.$$scope = { dirty: a, ctx: l }), e.$set(s);
    },
    i(l) {
      t || (W(e.$$.fragment, l), t = !0);
    },
    o(l) {
      Y(e.$$.fragment, l), t = !1;
    },
    d(l) {
      ne(e, l);
    }
  };
}
function we(o) {
  let e, t, l, a, s = (
    /*show_roi*/
    o[19] && Ie(o)
  );
  return {
    c() {
      e = v("img"), l = M(), s && s.c(), a = me(), this.h();
    },
    l(r) {
      e = p(r, "IMG", { src: !0, alt: !0, class: !0 }), l = F(r), s && s.l(r), a = me(), this.h();
    },
    h() {
      Ce(e.src, t = URL.createObjectURL(
        /*uploaded_file*/
        o[15]
      )) || R(e, "src", t), R(e, "alt", "Medical scan"), R(e, "class", "svelte-197pbtm");
    },
    m(r, u) {
      U(r, e, u), U(r, l, u), s && s.m(r, u), U(r, a, u);
    },
    p(r, u) {
      u[0] & /*uploaded_file*/
      32768 && !Ce(e.src, t = URL.createObjectURL(
        /*uploaded_file*/
        r[15]
      )) && R(e, "src", t), /*show_roi*/
      r[19] ? s ? s.p(r, u) : (s = Ie(r), s.c(), s.m(a.parentNode, a)) : s && (s.d(1), s = null);
    },
    d(r) {
      r && (h(e), h(l), h(a)), s && s.d(r);
    }
  };
}
function Ie(o) {
  let e;
  return {
    c() {
      e = v("div"), this.h();
    },
    l(t) {
      e = p(t, "DIV", { class: !0, style: !0 }), V(e).forEach(h), this.h();
    },
    h() {
      R(e, "class", "roi-marker svelte-197pbtm"), q(
        e,
        "left",
        /*roi*/
        o[18].x + "px"
      ), q(
        e,
        "top",
        /*roi*/
        o[18].y + "px"
      ), q(
        e,
        "width",
        /*roi*/
        o[18].radius * 2 + "px"
      ), q(
        e,
        "height",
        /*roi*/
        o[18].radius * 2 + "px"
      );
    },
    m(t, l) {
      U(t, e, l);
    },
    p(t, l) {
      l[0] & /*roi*/
      262144 && q(
        e,
        "left",
        /*roi*/
        t[18].x + "px"
      ), l[0] & /*roi*/
      262144 && q(
        e,
        "top",
        /*roi*/
        t[18].y + "px"
      ), l[0] & /*roi*/
      262144 && q(
        e,
        "width",
        /*roi*/
        t[18].radius * 2 + "px"
      ), l[0] & /*roi*/
      262144 && q(
        e,
        "height",
        /*roi*/
        t[18].radius * 2 + "px"
      );
    },
    d(t) {
      t && h(e);
    }
  };
}
function Oe(o) {
  let e, t;
  return {
    c() {
      e = v("div"), t = new Te(!1), this.h();
    },
    l(l) {
      e = p(l, "DIV", { class: !0 });
      var a = V(e);
      t = Se(a, !1), a.forEach(h), this.h();
    },
    h() {
      t.a = null, R(e, "class", "report-container svelte-197pbtm");
    },
    m(l, a) {
      U(l, e, a), t.m(
        /*visual_report*/
        o[17],
        e
      );
    },
    p(l, a) {
      a[0] & /*visual_report*/
      131072 && t.p(
        /*visual_report*/
        l[17]
      );
    },
    d(l) {
      l && h(e);
    }
  };
}
function Re(o) {
  let e, t, l = "JSON Output (for AI Agents)", a, s, r = JSON.stringify(
    /*analysis_results*/
    o[16],
    null,
    2
  ) + "", u;
  return {
    c() {
      e = v("details"), t = v("summary"), t.textContent = l, a = M(), s = v("pre"), u = pe(r), this.h();
    },
    l(m) {
      e = p(m, "DETAILS", { class: !0 });
      var I = V(e);
      t = p(I, "SUMMARY", { class: !0, "data-svelte-h": !0 }), j(t) !== "svelte-16bwjzd" && (t.textContent = l), a = F(I), s = p(I, "PRE", { class: !0 });
      var C = V(s);
      u = he(C, r), C.forEach(h), I.forEach(h), this.h();
    },
    h() {
      R(t, "class", "svelte-197pbtm"), R(s, "class", "svelte-197pbtm"), R(e, "class", "json-output svelte-197pbtm");
    },
    m(m, I) {
      U(m, e, I), g(e, t), g(e, a), g(e, s), g(s, u);
    },
    p(m, I) {
      I[0] & /*analysis_results*/
      65536 && r !== (r = JSON.stringify(
        /*analysis_results*/
        m[16],
        null,
        2
      ) + "") && qe(u, r);
    },
    d(m) {
      m && h(e);
    }
  };
}
function ze(o) {
  let e, t, l, a, s = "Supports: DICOM (.dcm), Images (.png, .jpg), and files without extensions (IM_0001, etc.)";
  return {
    c() {
      e = pe("Drop Medical Image File Here - or - Click to Upload"), t = v("br"), l = M(), a = v("span"), a.textContent = s, this.h();
    },
    l(r) {
      e = he(r, "Drop Medical Image File Here - or - Click to Upload"), t = p(r, "BR", {}), l = F(r), a = p(r, "SPAN", { style: !0, "data-svelte-h": !0 }), j(a) !== "svelte-l91joy" && (a.textContent = s), this.h();
    },
    h() {
      q(a, "font-size", "0.9em"), q(a, "color", "var(--body-text-color-subdued)");
    },
    m(r, u) {
      U(r, e, u), U(r, t, u), U(r, l, u), U(r, a, u);
    },
    p: Le,
    d(r) {
      r && (h(e), h(t), h(l), h(a));
    }
  };
}
function Be(o) {
  let e, t;
  return e = new UploadText({
    props: {
      i18n: (
        /*gradio*/
        o[13].i18n
      ),
      type: "file",
      $$slots: { default: [ze] },
      $$scope: { ctx: o }
    }
  }), {
    c() {
      le(e.$$.fragment);
    },
    l(l) {
      te(e.$$.fragment, l);
    },
    m(l, a) {
      ie(e, l, a), t = !0;
    },
    p(l, a) {
      const s = {};
      a[0] & /*gradio*/
      8192 && (s.i18n = /*gradio*/
      l[13].i18n), a[1] & /*$$scope*/
      4 && (s.$$scope = { dirty: a, ctx: l }), e.$set(s);
    },
    i(l) {
      t || (W(e.$$.fragment, l), t = !0);
    },
    o(l) {
      Y(e.$$.fragment, l), t = !1;
    },
    d(l) {
      ne(e, l);
    }
  };
}
function He(o) {
  let e, t, l, a, s, r, u, m;
  const I = [
    {
      autoscroll: (
        /*gradio*/
        o[13].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      o[13].i18n
    ) },
    /*loading_status*/
    o[9]
  ];
  let C = {};
  for (let c = 0; c < I.length; c += 1)
    C = Ne(C, I[c]);
  e = new StatusTracker({ props: C }), l = new BlockLabel({
    props: {
      show_label: (
        /*show_label*/
        o[7]
      ),
      Icon: Image,
      label: (
        /*label*/
        o[6] || "Medical Image Analyzer"
      )
    }
  });
  const z = [Xe, Ve], k = [];
  function B(c, b) {
    return (
      /*value*/
      c[0] === null || !/*uploaded_file*/
      c[15] ? 0 : 1
    );
  }
  return s = B(o), r = k[s] = z[s](o), {
    c() {
      le(e.$$.fragment), t = M(), le(l.$$.fragment), a = M(), r.c(), u = me();
    },
    l(c) {
      te(e.$$.fragment, c), t = F(c), te(l.$$.fragment, c), a = F(c), r.l(c), u = me();
    },
    m(c, b) {
      ie(e, c, b), U(c, t, b), ie(l, c, b), U(c, a, b), k[s].m(c, b), U(c, u, b), m = !0;
    },
    p(c, b) {
      const S = b[0] & /*gradio, loading_status*/
      8704 ? Ae(I, [
        b[0] & /*gradio*/
        8192 && {
          autoscroll: (
            /*gradio*/
            c[13].autoscroll
          )
        },
        b[0] & /*gradio*/
        8192 && { i18n: (
          /*gradio*/
          c[13].i18n
        ) },
        b[0] & /*loading_status*/
        512 && De(
          /*loading_status*/
          c[9]
        )
      ]) : {};
      e.$set(S);
      const w = {};
      b[0] & /*show_label*/
      128 && (w.show_label = /*show_label*/
      c[7]), b[0] & /*label*/
      64 && (w.label = /*label*/
      c[6] || "Medical Image Analyzer"), l.$set(w);
      let O = s;
      s = B(c), s === O ? k[s].p(c, b) : (Fe(), Y(k[O], 1, 1, () => {
        k[O] = null;
      }), Ue(), r = k[s], r ? r.p(c, b) : (r = k[s] = z[s](c), r.c()), W(r, 1), r.m(u.parentNode, u));
    },
    i(c) {
      m || (W(e.$$.fragment, c), W(l.$$.fragment, c), W(r), m = !0);
    },
    o(c) {
      Y(e.$$.fragment, c), Y(l.$$.fragment, c), Y(r), m = !1;
    },
    d(c) {
      c && (h(t), h(a), h(u)), ne(e, c), ne(l, c), k[s].d(c);
    }
  };
}
function Je(o) {
  let e, t;
  return e = new Block({
    props: {
      visible: (
        /*visible*/
        o[5]
      ),
      elem_id: (
        /*elem_id*/
        o[3]
      ),
      elem_classes: (
        /*elem_classes*/
        o[4]
      ),
      container: (
        /*container*/
        o[10]
      ),
      scale: (
        /*scale*/
        o[11]
      ),
      min_width: (
        /*min_width*/
        o[12]
      ),
      allow_overflow: !1,
      padding: !0,
      $$slots: { default: [He] },
      $$scope: { ctx: o }
    }
  }), {
    c() {
      le(e.$$.fragment);
    },
    l(l) {
      te(e.$$.fragment, l);
    },
    m(l, a) {
      ie(e, l, a), t = !0;
    },
    p(l, a) {
      const s = {};
      a[0] & /*visible*/
      32 && (s.visible = /*visible*/
      l[5]), a[0] & /*elem_id*/
      8 && (s.elem_id = /*elem_id*/
      l[3]), a[0] & /*elem_classes*/
      16 && (s.elem_classes = /*elem_classes*/
      l[4]), a[0] & /*container*/
      1024 && (s.container = /*container*/
      l[10]), a[0] & /*scale*/
      2048 && (s.scale = /*scale*/
      l[11]), a[0] & /*min_width*/
      4096 && (s.min_width = /*min_width*/
      l[12]), a[0] & /*root, gradio, value, uploaded_file, analysis_results, analysis_mode, visual_report, roi, show_roi, task, modality, show_label, label, loading_status*/
      1041351 | a[1] & /*$$scope*/
      4 && (s.$$scope = { dirty: a, ctx: l }), e.$set(s);
    },
    i(l) {
      t || (W(e.$$.fragment, l), t = !0);
    },
    o(l) {
      Y(e.$$.fragment, l), t = !1;
    },
    d(l) {
      ne(e, l);
    }
  };
}
let Qe = !1;
function We(o, e, t) {
  let { elem_id: l = "" } = e, { elem_classes: a = [] } = e, { visible: s = !0 } = e, { value: r = null } = e, { label: u } = e, { show_label: m } = e, { show_download_button: I } = e, { root: C } = e, { proxy_url: z } = e, { loading_status: k } = e, { container: B = !0 } = e, { scale: c = null } = e, { min_width: b = void 0 } = e, { gradio: S } = e, { analysis_mode: w = "structured" } = e, { include_confidence: O = !0 } = e, { include_reasoning: Z = !0 } = e, { modality: T = "CT" } = e, { task: G = "full_analysis" } = e, E = null, X = { x: 256, y: 256, radius: 10 }, H = !1, D = null, A = "";
  async function se(n) {
    var _;
    const i = URL.createObjectURL(n), f = ((_ = n.name.split(".").pop()) == null ? void 0 : _.toLowerCase()) || "";
    try {
      if (!f || f === "dcm" || f === "dicom" || n.type === "application/dicom" || n.name.startsWith("IM_")) {
        const d = new FormData();
        d.append("file", n);
        const N = await fetch(`${C}/process_dicom`, { method: "POST", body: d });
        if (N.ok)
          return await N.json();
      }
      return {
        url: i,
        name: n.name,
        size: n.size,
        type: n.type || "application/octet-stream"
      };
    } catch (d) {
      throw console.error("Error loading file:", d), d;
    }
  }
  function ae({ detail: n }) {
    const i = n;
    se(i).then((f) => {
      t(15, E = i), S.dispatch && S.dispatch("upload", { file: i, data: f });
    }).catch((f) => {
      console.error("Upload error:", f);
    });
  }
  function L() {
    t(0, r = null), t(15, E = null), t(16, D = null), t(17, A = ""), S.dispatch("clear");
  }
  function oe(n) {
    if (!H) return;
    const i = n.target.getBoundingClientRect();
    t(18, X.x = Math.round(n.clientX - i.left), X), t(18, X.y = Math.round(n.clientY - i.top), X), S.dispatch && S.dispatch("change", { roi: X });
  }
  function x(n) {
    var f, _, d, N, P, K, ee, ce, ve;
    if (!n) return "";
    let i = '<div class="medical-report">';
    if (i += "<h3>🏥 Medical Image Analysis Report</h3>", i += '<div class="report-section">', i += "<h4>📋 Basic Information</h4>", i += `<p><strong>Modality:</strong> ${n.modality || "Unknown"}</p>`, i += `<p><strong>Timestamp:</strong> ${n.timestamp || "N/A"}</p>`, i += "</div>", n.point_analysis) {
      const y = n.point_analysis;
      i += '<div class="report-section">', i += "<h4>🎯 Point Analysis</h4>", i += `<p><strong>Location:</strong> (${(f = y.location) == null ? void 0 : f.x}, ${(_ = y.location) == null ? void 0 : _.y})</p>`, n.modality === "CT" ? i += `<p><strong>HU Value:</strong> ${((d = y.hu_value) == null ? void 0 : d.toFixed(1)) || "N/A"}</p>` : i += `<p><strong>Intensity:</strong> ${((N = y.intensity) == null ? void 0 : N.toFixed(3)) || "N/A"}</p>`, y.tissue_type && (i += `<p><strong>Tissue Type:</strong> ${y.tissue_type.icon || ""} ${y.tissue_type.type || "Unknown"}</p>`), O && y.confidence !== void 0 && (i += `<p><strong>Confidence:</strong> ${y.confidence}</p>`), Z && y.reasoning && (i += `<p class="reasoning">💭 ${y.reasoning}</p>`), i += "</div>";
    }
    if ((P = n.segmentation) != null && P.statistics) {
      const y = n.segmentation.statistics;
      if (n.modality === "CT" && y.total_fat_percentage !== void 0) {
        if (i += '<div class="report-section">', i += "<h4>🔬 Fat Segmentation</h4>", i += '<div class="stats-grid">', i += `<div><strong>Total Fat:</strong> ${y.total_fat_percentage.toFixed(1)}%</div>`, i += `<div><strong>Subcutaneous:</strong> ${y.subcutaneous_fat_percentage.toFixed(1)}%</div>`, i += `<div><strong>Visceral:</strong> ${y.visceral_fat_percentage.toFixed(1)}%</div>`, i += `<div><strong>V/S Ratio:</strong> ${y.visceral_subcutaneous_ratio.toFixed(2)}</div>`, i += "</div>", n.segmentation.interpretation) {
          const J = n.segmentation.interpretation;
          i += '<div class="interpretation">', i += `<p><strong>Obesity Risk:</strong> <span class="risk-${J.obesity_risk}">${J.obesity_risk.toUpperCase()}</span></p>`, i += `<p><strong>Visceral Risk:</strong> <span class="risk-${J.visceral_risk}">${J.visceral_risk.toUpperCase()}</span></p>`, ((K = J.recommendations) == null ? void 0 : K.length) > 0 && (i += "<p><strong>Recommendations:</strong></p>", i += "<ul>", J.recommendations.forEach((ge) => {
            i += `<li>${ge}</li>`;
          }), i += "</ul>"), i += "</div>";
        }
        i += "</div>";
      } else if (n.segmentation.tissue_distribution) {
        i += '<div class="report-section">', i += "<h4>🦴 Tissue Distribution</h4>", i += '<div class="tissue-grid">';
        const J = n.segmentation.tissue_distribution, ge = {
          bone: "🦴",
          soft_tissue: "🔴",
          air: "🌫️",
          metal: "⚙️",
          fat: "🟡",
          fluid: "💧"
        };
        Object.entries(J).forEach(([ue, be]) => {
          be > 0 && (i += '<div class="tissue-item">', i += `<div class="tissue-icon">${ge[ue] || "📍"}</div>`, i += `<div class="tissue-name">${ue.replace("_", " ")}</div>`, i += `<div class="tissue-percentage">${be.toFixed(1)}%</div>`, i += "</div>");
        }), i += "</div>", ((ee = n.segmentation.clinical_findings) == null ? void 0 : ee.length) > 0 && (i += '<div class="clinical-findings">', i += "<p><strong>⚠️ Clinical Findings:</strong></p>", i += "<ul>", n.segmentation.clinical_findings.forEach((ue) => {
          i += `<li>${ue.description} (Confidence: ${ue.confidence})</li>`;
        }), i += "</ul>", i += "</div>"), i += "</div>";
      }
    }
    if (n.quality_metrics) {
      const y = n.quality_metrics;
      i += '<div class="report-section">', i += "<h4>📊 Image Quality</h4>", i += `<p><strong>Overall Quality:</strong> <span class="quality-${y.overall_quality}">${((ce = y.overall_quality) == null ? void 0 : ce.toUpperCase()) || "UNKNOWN"}</span></p>`, ((ve = y.issues) == null ? void 0 : ve.length) > 0 && (i += `<p><strong>Issues:</strong> ${y.issues.join(", ")}</p>`), i += "</div>";
    }
    return i += "</div>", i;
  }
  function $() {
    T = ke(this), t(1, T);
  }
  function re() {
    G = ke(this), t(2, G);
  }
  function _e() {
    H = this.checked, t(19, H);
  }
  return o.$$set = (n) => {
    "elem_id" in n && t(3, l = n.elem_id), "elem_classes" in n && t(4, a = n.elem_classes), "visible" in n && t(5, s = n.visible), "value" in n && t(0, r = n.value), "label" in n && t(6, u = n.label), "show_label" in n && t(7, m = n.show_label), "show_download_button" in n && t(23, I = n.show_download_button), "root" in n && t(8, C = n.root), "proxy_url" in n && t(24, z = n.proxy_url), "loading_status" in n && t(9, k = n.loading_status), "container" in n && t(10, B = n.container), "scale" in n && t(11, c = n.scale), "min_width" in n && t(12, b = n.min_width), "gradio" in n && t(13, S = n.gradio), "analysis_mode" in n && t(14, w = n.analysis_mode), "include_confidence" in n && t(25, O = n.include_confidence), "include_reasoning" in n && t(26, Z = n.include_reasoning), "modality" in n && t(1, T = n.modality), "task" in n && t(2, G = n.task);
  }, o.$$.update = () => {
    o.$$.dirty[0] & /*analysis_results*/
    65536 && D && t(17, A = x(D)), o.$$.dirty[0] & /*uploaded_file, analysis_results, visual_report*/
    229376 && t(0, r = {
      image: E,
      analysis: D,
      report: A
    });
  }, [
    r,
    T,
    G,
    l,
    a,
    s,
    u,
    m,
    C,
    k,
    B,
    c,
    b,
    S,
    w,
    E,
    D,
    A,
    X,
    H,
    ae,
    L,
    oe,
    I,
    z,
    O,
    Z,
    $,
    re,
    _e
  ];
}
class Ye extends Ee {
  constructor(e) {
    super(), Me(
      this,
      e,
      We,
      Je,
      je,
      {
        elem_id: 3,
        elem_classes: 4,
        visible: 5,
        value: 0,
        label: 6,
        show_label: 7,
        show_download_button: 23,
        root: 8,
        proxy_url: 24,
        loading_status: 9,
        container: 10,
        scale: 11,
        min_width: 12,
        gradio: 13,
        analysis_mode: 14,
        include_confidence: 25,
        include_reasoning: 26,
        modality: 1,
        task: 2
      },
      null,
      [-1, -1]
    );
  }
}
export {
  Ye as default
};
