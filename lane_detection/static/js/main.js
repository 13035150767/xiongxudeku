(function () {
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const uploadArea = $("#uploadArea");
    const fileInput = $("#fileInput");
    const fileInfo = $("#fileInfo");
    const fileName = $("#fileName");
    const clearFile = $("#clearFile");
    const detectBtn = $("#detectBtn");
    const btnSpinner = $("#btnSpinner");
    const btnText = $("#btnText");
    const placeholder = $("#placeholder");
    const resultsGrid = $("#resultsGrid");
    const yoloResult = $("#yoloResult");
    const videoResult = $("#videoResult");
    const toast = $("#toast");
    const laneParams = $("#laneParams");
    const yoloParams = $("#yoloParams");
    const pipelineLane = $("#pipelineLane");
    const pipelineYolo = $("#pipelineYolo");

    let selectedFile = null;
    let isProcessing = false;
    let currentMode = "lane";

    const sliderMap = {
        cannyThreshold1: { display: "canny1Val", param: "canny_threshold1" },
        cannyThreshold2: { display: "canny2Val", param: "canny_threshold2" },
        gaussianKernel: { display: "gaussVal", param: "gaussian_kernel" },
        houghThreshold: { display: "houghThVal", param: "hough_threshold" },
        houghMinLineLength: { display: "houghMinVal", param: "hough_min_line_length" },
        houghMaxLineGap: { display: "houghGapVal", param: "hough_max_line_gap" },
    };

    Object.keys(sliderMap).forEach(function (id) {
        var el = $("#" + id);
        var display = $("#" + sliderMap[id].display);
        el.addEventListener("input", function () {
            var val = parseInt(el.value);
            if (id === "gaussianKernel" && val % 2 === 0) { val += 1; el.value = val; }
            display.textContent = val;
        });
    });

    $("#confThreshold").addEventListener("input", function () {
        $("#confVal").textContent = this.value;
    });

    $$(".mode-tab").forEach(function (tab) {
        tab.addEventListener("click", function () {
            $$(".mode-tab").forEach(function (t) { t.classList.remove("active"); });
            tab.classList.add("active");
            currentMode = tab.dataset.mode;

            var isYolo = currentMode !== "lane";
            laneParams.style.display = isYolo ? "none" : "block";
            yoloParams.style.display = isYolo ? "block" : "none";
            pipelineLane.style.display = isYolo ? "none" : "flex";
            pipelineYolo.style.display = isYolo ? "flex" : "none";

            hideAllResults();
        });
    });

    function hideAllResults() {
        placeholder.style.display = "flex";
        resultsGrid.style.display = "none";
        yoloResult.style.display = "none";
        videoResult.style.display = "none";
    }

    uploadArea.addEventListener("click", function () { fileInput.click(); });
    uploadArea.addEventListener("dragover", function (e) { e.preventDefault(); uploadArea.classList.add("dragover"); });
    uploadArea.addEventListener("dragleave", function () { uploadArea.classList.remove("dragover"); });
    uploadArea.addEventListener("drop", function (e) {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });
    clearFile.addEventListener("click", function () {
        selectedFile = null;
        fileInput.value = "";
        fileInfo.style.display = "none";
        detectBtn.disabled = true;
    });

    function handleFile(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileInfo.style.display = "flex";
        detectBtn.disabled = false;
    }

    detectBtn.addEventListener("click", async function () {
        if (!selectedFile || isProcessing) return;
        isProcessing = true;
        detectBtn.disabled = true;
        btnSpinner.style.display = "block";
        btnText.textContent = "检测中...";

        var isVideo = selectedFile.type.startsWith("video/");
        try {
            if (currentMode === "lane") {
                if (isVideo) await processLaneVideo();
                else await processLaneImage();
            } else if (currentMode === "traffic") {
                if (isVideo) await processYoloVideo("/api/segment", "分割车流量");
                else await processYoloImage("/api/segment", "分割车流量", renderSegTrafficStats);
            } else if (currentMode === "speed") {
                if (isVideo) await processYoloVideo("/api/speed_detect", "车速检测");
                else await processYoloImage("/api/speed_detect", "车速检测", renderSpeedStats);
            } else if (currentMode === "heatmap") {
                if (isVideo) await processYoloVideo("/api/heatmap", "热力图");
                else await processYoloImage("/api/heatmap", "热力图", renderHeatmapStats);
            } else if (currentMode === "lane_occupancy") {
                if (isVideo) await processYoloVideo("/api/lane_occupancy", "车道占用");
                else await processYoloImage("/api/lane_occupancy", "车道占用", renderLaneOccupancyStats);
            }
        } catch (err) {
            showToast("处理失败: " + err.message, "error");
        } finally {
            isProcessing = false;
            detectBtn.disabled = false;
            btnSpinner.style.display = "none";
            btnText.textContent = "开始检测";
        }
    });

    function getLaneFormData() {
        var formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("use_hsl", $("#useHsl").value);
        Object.keys(sliderMap).forEach(function (id) {
            formData.append(sliderMap[id].param, $("#" + id).value);
        });
        return formData;
    }

    function getYoloFormData() {
        var formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("conf", $("#confThreshold").value);
        return formData;
    }

    async function processLaneImage() {
        var formData = getLaneFormData();
        var resp = await fetch("/api/detect_image", { method: "POST", body: formData });
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "未知错误");

        placeholder.style.display = "none";
        yoloResult.style.display = "none";
        videoResult.style.display = "none";
        resultsGrid.style.display = "grid";

        var stepKeys = ["original", "color_filter", "grayscale", "gaussian_blur", "canny_edge", "roi", "hough_lines", "result"];
        stepKeys.forEach(function (key) {
            var img = $("#step_" + key);
            if (data.steps[key]) img.src = "data:image/jpeg;base64," + data.steps[key];
        });
        activatePipelineSteps(stepKeys);
        showToast("车道线检测完成！", "success");
    }

    async function processLaneVideo() {
        var formData = getLaneFormData();
        showToast("正在处理视频，请稍候...", "info");
        var resp = await fetch("/api/detect_video", { method: "POST", body: formData });
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "未知错误");

        placeholder.style.display = "none";
        resultsGrid.style.display = "none";
        yoloResult.style.display = "none";
        videoResult.style.display = "block";
        $("#videoStatsBar").innerHTML = "";
        $("#resultVideo").src = data.video_url;
        $("#videoInfo").textContent = "共处理 " + data.processed_frames + " 帧";
        showToast("视频处理完成！", "success");
    }

    async function processYoloImage(endpoint, title, statsRenderer) {
        var formData = getYoloFormData();
        var resp = await fetch(endpoint, { method: "POST", body: formData });
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "未知错误");

        placeholder.style.display = "none";
        resultsGrid.style.display = "none";
        videoResult.style.display = "none";
        yoloResult.style.display = "block";

        $("#yoloResultLabel").textContent = title;
        $("#yoloResultTitle").textContent = "";
        if (data.image) {
            $("#yoloResultImg").src = "data:image/jpeg;base64," + data.image;
        }

        if (data.stats && statsRenderer) {
            statsRenderer(data.stats);
        } else {
            $("#statsBar").innerHTML = "";
        }

        showToast(title + "完成！", "success");
    }

    async function processYoloVideo(endpoint, title) {
        var formData = getYoloFormData();
        showToast("正在处理视频，请稍候...", "info");
        var resp = await fetch(endpoint, { method: "POST", body: formData });
        var data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "未知错误");

        placeholder.style.display = "none";
        resultsGrid.style.display = "none";
        yoloResult.style.display = "none";
        videoResult.style.display = "block";

        var statsHtml = "";
        if (data.stats) {
            if (data.stats.total_count !== undefined) {
                statsHtml += '<div class="stat-item"><span class="stat-label">车流量</span><span class="stat-value">' + data.stats.total_count + '</span></div>';
            }
            if (data.stats.avg_speed !== undefined) {
                statsHtml += '<div class="stat-item"><span class="stat-label">平均车速</span><span class="stat-value">' + data.stats.avg_speed + ' km/h</span></div>';
            }
        }
        $("#videoStatsBar").innerHTML = statsHtml;
        $("#resultVideo").src = data.video_url;
        $("#videoInfo").textContent = "共处理 " + data.processed_frames + " 帧";
        showToast(title + "视频处理完成！", "success");
    }

    function renderSegTrafficStats(stats) {
        var html = '<div class="stat-item"><span class="stat-label">检测车辆</span><span class="stat-value">' + stats.vehicle_count + '</span></div>';
        html += '<div class="stat-item"><span class="stat-label">车流量</span><span class="stat-value" style="color:#22d3ee">' + (stats.total_count || 0) + '</span></div>';
        if (stats.by_type) {
            Object.keys(stats.by_type).forEach(function (type) {
                html += '<div class="stat-item"><span class="stat-label">' + type + '</span><span class="stat-value">' + stats.by_type[type] + '</span></div>';
            });
        }
        $("#statsBar").innerHTML = html;
    }

    function renderSpeedStats(stats) {
        var html = '<div class="stat-item"><span class="stat-label">检测车辆</span><span class="stat-value">' + stats.vehicle_count + '</span></div>';
        html += '<div class="stat-item"><span class="stat-label">平均车速</span><span class="stat-value">' + stats.avg_speed + ' km/h</span></div>';
        if (stats.speeds) {
            var ids = Object.keys(stats.speeds);
            ids.forEach(function (id) {
                var spd = stats.speeds[id];
                var cls = spd < 40 ? "speed-low" : spd < 60 ? "speed-mid" : "speed-high";
                html += '<div class="stat-item"><span class="stat-label">车辆 #' + id + '</span><span class="stat-value ' + cls + '">' + spd + ' km/h</span></div>';
            });
        }
        $("#statsBar").innerHTML = html;
    }

    function renderHeatmapStats(stats) {
        var html = '<div class="stat-item"><span class="stat-label">检测车辆</span><span class="stat-value">' + stats.vehicle_count + '</span></div>';
        html += '<div class="stat-item"><span class="stat-label">最大热值</span><span class="stat-value">' + stats.max_intensity + '</span></div>';
        html += '<div class="stat-item"><span class="stat-label">平均热值</span><span class="stat-value">' + stats.avg_intensity + '</span></div>';
        $("#statsBar").innerHTML = html;
    }

    function renderLaneOccupancyStats(stats) {
        var html = '<div class="stat-item"><span class="stat-label">总车辆</span><span class="stat-value">' + stats.total_vehicles + '</span></div>';
        if (stats.lanes) {
            stats.lanes.forEach(function (lane) {
                var color = lane.status === "FREE" ? "#34d399" : lane.status === "NORMAL" ? "#22d3ee" : lane.status === "SLOW" ? "#fbbf24" : "#f87171";
                html += '<div class="stat-item"><span class="stat-label">车道' + lane.lane + ' (' + lane.vehicle_count + '辆)</span><span class="stat-value" style="color:' + color + '">' + lane.status + '</span></div>';
            });
        }
        $("#statsBar").innerHTML = html;
    }

    function activatePipelineSteps(stepKeys) {
        $$(".pipeline-step").forEach(function (el) {
            var step = el.dataset.step;
            if (stepKeys && stepKeys.indexOf(step) >= 0) {
                el.classList.add("active");
            } else {
                el.classList.remove("active");
            }
        });
    }

    $$(".result-card").forEach(function (card) {
        card.addEventListener("click", function () {
            $$(".result-card").forEach(function (c) { c.classList.remove("active"); });
            card.classList.add("active");
        });
    });

    $$(".pipeline-step").forEach(function (step) {
        step.addEventListener("click", function () {
            var stepName = step.dataset.step;
            var card = document.querySelector('.result-card[data-step="' + stepName + '"]');
            if (card) {
                $$(".result-card").forEach(function (c) { c.classList.remove("active"); });
                card.classList.add("active");
                card.scrollIntoView({ behavior: "smooth", block: "center" });
            }
        });
    });

    function showToast(message, type) {
        toast.textContent = message;
        toast.className = "toast " + type + " show";
        setTimeout(function () { toast.classList.remove("show"); }, 3000);
    }
})();
