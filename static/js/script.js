document.addEventListener("DOMContentLoaded", () => {
  const els = {
    imageInput: document.getElementById("imageInput"),
    uploadedImage: document.getElementById("uploadedImage"),
    canvas: document.getElementById("canvasOutput"),
    detectBtn: document.getElementById("detectImage"),
    spinner: document.getElementById("loadingSpinner"),
    btnText: document.getElementById("buttonText"),
    modelSelect: document.getElementById("modelSelect"),
    signDetails: document.getElementById("signDetails"),
    noDetection: document.getElementById("noDetectionMessage"),
    confidenceChartEl: document.getElementById("confidenceChart"),
    detectingRing: document.getElementById("detectingRing"),
  };

  let confidenceChart = null;

  const toggleLoading = (isLoading) => {
    els.spinner.classList.toggle("hidden", !isLoading);
    els.btnText.textContent = isLoading ? "Detecting..." : "Detect Signs";
    els.detectBtn.disabled = isLoading;
    els.detectingRing.classList.toggle("hidden", !isLoading);
  };

  const resizeCanvasToImage = () => {
    // Make canvas match the displayed image size and position it over the image
    const imgRect = els.uploadedImage.getBoundingClientRect();
    const parentRect = els.uploadedImage.parentElement.getBoundingClientRect();

    // set width/height in CSS pixels
    els.canvas.width = els.uploadedImage.clientWidth;
    els.canvas.height = els.uploadedImage.clientHeight;

    // position canvas relative to the image container
    els.canvas.style.position = "absolute";
    // compute left/top relative to container (container is .image-container, which is positioned relative)
    const offsetLeft = els.uploadedImage.offsetLeft;
    const offsetTop = els.uploadedImage.offsetTop;
    els.canvas.style.left = `${offsetLeft}px`;
    els.canvas.style.top = `${offsetTop}px`;
    els.canvas.style.transform = "none";
    els.canvas.style.pointerEvents = "none";
  };

  const scaleDetections = (dets) => {
    const iw = els.uploadedImage.naturalWidth || els.uploadedImage.width;
    const ih = els.uploadedImage.naturalHeight || els.uploadedImage.height;
    const cw = els.canvas.width,
      ch = els.canvas.height;
    if (!iw || !ih || !cw || !ch) return dets;
    const sx = cw / iw,
      sy = ch / ih;
    return dets.map((d) => {
      const scaled = d.bbox.map((v, i) => {
        return i % 2 === 0 ? Math.round(v * sx) : Math.round(v * sy);
      });
      return { ...d, bbox: scaled };
    });
  };

  const drawDetections = (dets) => {
    const ctx = els.canvas.getContext("2d");
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

    ctx.lineWidth = 2;
    ctx.font = "14px Arial";
    ctx.textBaseline = "top";

    dets.forEach((d, idx) => {
      const [x1, y1, x2, y2] = d.bbox;
      // color by index for differentiation
      const hue = (idx * 47) % 360;
      ctx.strokeStyle = `hsl(${hue} 80% 40%)`;
      ctx.fillStyle = `hsl(${hue} 80% 40%)`;
      ctx.beginPath();
      ctx.rect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
      ctx.stroke();

      const text = `${d.details?.name || d.class_name || "Unknown"} ${(
        d.confidence * 100
      ).toFixed(1)}%`;
      // measure text background
      const metrics = ctx.measureText(text);
      const padding = 6;
      const textW = metrics.width + padding;
      const textH = 18 + 4; // approx
      // background
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(x1, Math.max(0, y1 - textH), textW, textH);
      // text
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x1 + 4, Math.max(0, y1 - textH) + 4);
    });
  };

  const updateDetails = (dets) => {
    if (!dets || dets.length === 0) {
      els.noDetection.classList.remove("hidden");
      els.signDetails.innerHTML =
        '<p class="text-gray-500 italic text-center">No signs detected.</p>';
      updateChart([], []);
      return;
    }

    els.noDetection.classList.add("hidden");

    els.signDetails.innerHTML = dets
      .map((d) => {
        const detail = d.details || {};
        return `
            <div class="sign-detail mb-6 p-5 bg-white rounded-xl border border-blue-200 shadow-sm">
                <div class="flex items-start">
                    <div class="flex-1">
                        <h4 class="text-xl font-bold text-blue-800 mb-2">${
                          detail.name || d.class_name
                        }</h4>
                        <p class="text-gray-700 mb-1"><span class="font-semibold">Mã biển:</span> ${
                          detail.code || d.class_name
                        }</p>
                        <p class="text-gray-700 mb-3"><span class="font-semibold">Mô tả:</span> ${
                          detail.description || "No description available"
                        }</p>
                        <div class="flex items-center">
                            <span class="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                                Độ tin cậy: ${(d.confidence * 100).toFixed(1)}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            `;
      })
      .join("");

    // Update chart
    const labels = dets.map((d) => d.details?.name || d.class_name);
    const values = dets.map((d) => +(d.confidence || 0).toFixed(3));
    updateChart(labels, values);
  };

  const updateChart = (labels, values) => {
    if (confidenceChart) {
      try {
        confidenceChart.destroy();
      } catch (e) {}
      confidenceChart = null;
    }

    // đảm bảo canvas có chiều cao (style) - fallback nếu chưa set ở HTML
    if (!els.confidenceChartEl.style.height) {
      els.confidenceChartEl.style.height = "300px";
    }

    // nếu không có dữ liệu -> chỉ clear canvas và return
    if (!labels || labels.length === 0) {
      const ctx = els.confidenceChartEl.getContext("2d");
      ctx.clearRect(
        0,
        0,
        els.confidenceChartEl.width,
        els.confidenceChartEl.height
      );
      return;
    }

    // Tạo chart mới
    confidenceChart = new Chart(els.confidenceChartEl.getContext("2d"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Confidence",
            data: values,
            backgroundColor: labels.map(() => "rgba(59,130,246,0.7)"),
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false, // canvas sẽ tuân theo kích thước CSS đã đặt
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${(ctx.raw * 100).toFixed(1)}%`,
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
            ticks: {
              callback: (val) => `${Math.round(val * 100)}%`,
            },
          },
        },
      },
    });
  };

  // =========================== CHANGE ===========================
  els.imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      els.uploadedImage.src = ev.target.result;
      els.uploadedImage.classList.remove("hidden");
      els.uploadedImage.onload = () => {
        resizeCanvasToImage();
        drawDetections([]);
        updateDetails([]);
      };
      els.detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  });
  // =========================== CLICK ===========================
  els.detectBtn.addEventListener("click", async () => {
    const file = els.imageInput.files[0];
    if (!file) return alert("Please select an image.");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", els.modelSelect.value);

    try {
      toggleLoading(true);
      await new Promise((resolve) => setTimeout(resolve, 50));

      const response = await fetch("/image_upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}: ${response.statusText}`
        );
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Xử lý khi thành công
      els.uploadedImage.src = data.image_url;

      // Đợi ảnh load
      await new Promise((resolve) => {
        els.uploadedImage.onload = resolve;
        els.uploadedImage.onerror = () => {
          throw new Error("Failed to load processed image");
        };
      });

      resizeCanvasToImage();
      const scaled = scaleDetections(data.detections || []);
      drawDetections(scaled);
      updateDetails(scaled); // Hàm này đã gọi updateChart()
    } catch (err) {
      console.error("Detection error:", err);
      const errorMessage = err.message.includes("Failed to fetch")
        ? "Network error - Please check your connection"
        : err.message || "An unknown error occurred";
      alert(`Detection error: ${errorMessage}`);
    } finally {
      toggleLoading(false);
    }
  });
});
