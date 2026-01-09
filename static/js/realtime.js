document.addEventListener("DOMContentLoaded", () => {
    const videoFeed = document.getElementById("videoFeed");
    const cameraPlaceholder = document.getElementById("cameraPlaceholder");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const detectionsLog = document.getElementById("detectionsLog");
    let isDetecting = false;
    let lastDetections = [];

    // Hiệu ứng khi bật/tắt camera
    const toggleCameraUI = (isOn) => {
        if (isOn) {
            videoFeed.classList.remove("opacity-0");
            cameraPlaceholder.classList.add("opacity-0");
            startBtn.classList.add("bg-green-600", "hover:bg-green-700");
            startBtn.classList.remove("bg-blue-600", "hover:bg-blue-700");
        } else {
            videoFeed.classList.add("opacity-0");
            cameraPlaceholder.classList.remove("opacity-0");
            startBtn.classList.remove("bg-green-600", "hover:bg-green-700");
            startBtn.classList.add("bg-blue-600", "hover:bg-blue-700");
        }
    };

    startBtn.addEventListener("click", () => {
        if (!isDetecting) {
            videoFeed.src = "/video_feed";
            isDetecting = true;
            toggleCameraUI(true);
            
            // Thêm hiệu ứng khi bật camera
            videoFeed.oncanplay = () => {
                videoFeed.play();
                cameraPlaceholder.style.transition = "opacity 0.5s ease";
            };
        }
    });

    stopBtn.addEventListener("click", () => {
        if (isDetecting) {
            videoFeed.src = "";
            isDetecting = false;
            toggleCameraUI(false);
            
            // Thêm hiệu ứng khi tắt camera
            cameraPlaceholder.style.transition = "opacity 0.5s ease";
        }
    });

    // Khởi tạo trạng thái ban đầu
    toggleCameraUI(false);

    const updateDetectionLog = async () => {
        if (!isDetecting) return;
        try {
            const response = await fetch("/api/history");
            if (!response.ok) throw new Error("Failed to fetch history");
            const detections = await response.json();
            if (JSON.stringify(detections) === JSON.stringify(lastDetections)) return;

            detectionsLog.innerHTML = detections
                .map(
                    (det, index) => `
                    <div class="detection-item p-3 bg-white rounded-lg mb-2 shadow-sm flex items-center">
                        <div class="w-8 h-8 rounded-full bg-blue-100 text-blue-800 flex items-center justify-center mr-3 font-bold">
                            ${index + 1}
                        </div>
                        <div>
                            <div class="font-semibold text-blue-800">${det.type || 'Unknown'}</div>
                            <div class="text-sm text-gray-500">${det.timestamp || new Date().toLocaleTimeString()}</div>
                        </div>
                        <div class="ml-auto px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                            ${det.confidence ? (det.confidence * 100).toFixed(1) + '%' : ''}
                        </div>
                    </div>
                `
                )
                .slice(0, 10)
                .join("");
            lastDetections = detections;
        } catch (err) {
            console.error("Error fetching detection log:", err);
        }
    };

    setInterval(updateDetectionLog, 1000);
});