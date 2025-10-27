// Global state
let currentJobId = null;
let optimizedLayoutData = null;
let currentViewMode = 'side-by-side';

document.addEventListener('DOMContentLoaded', async () => {
    const furnitureList = document.getElementById('furniture-list');
    const architecturalList = document.getElementById('architectural-list');
    const canvas = document.getElementById('canvas');
    const optimizedCanvas = document.getElementById('optimized-canvas');
    const deleteBtn = document.getElementById('delete-btn');
    const saveBtn = document.getElementById('save-btn');
    const optimizeBtn = document.getElementById('optimize-btn');
    const toggleDimsSwitch = document.getElementById('toggle-dims-switch');
    const loadBtn = document.getElementById('load-btn');
    const loadLayoutInput = document.getElementById('load-layout-input');
    const contextualSettings = document.getElementById('contextual-settings');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar');
    const closeSidebarBtn = document.getElementById('close-sidebar');
    const floatingSidebar = document.getElementById('floating-sidebar');
    
    // View toggle buttons
    const viewToggleContainer = document.getElementById('view-toggle-container');
    const viewSideBySideBtn = document.getElementById('view-side-by-side');
    
    let roomContainer;
    let optimizedRoomContainer;
    let selectedObject = null;
    let objectCounter = 0;
    const furnitureRatios = {};
    const furnitureBounds = {};

    const furnitureCatalog = [
        { name: 'Bed', image: '/static/images/bed.png', width: 160, height: 200, zHeight: 55 },
        { name: 'Study Table', image: '/static/images/study_table.png', width: 120, height: 80, zHeight: 75 },
        { name: 'Sofa', image: '/static/images/sofa.png', width: 160, height: 110, zHeight: 85 },
        { name: 'Wardrobe', image: '/static/images/wardrobe.png', width: 150, height: 60, zHeight: 200 },
        { name: 'Study Chair', image: '/static/images/study_chair.png', width: 50, height: 50, zHeight: 42 },
        { name: 'Bedside Table', image: '/static/images/bedside_table.png', width: 60, height: 60, zHeight: 60 }
    ];

    const architecturalCatalog = [
        { name: 'Door', type: 'door', image: '/static/images/door.png', width: 90, openingHeight: 210 },
        { name: 'Window', type: 'window', image: '/static/images/window.png', width: 120, openingHeight: 100 }
    ];

    // ==================== MODEL LOADING ====================
    
    async function checkModelStatus() {
        try {
            const response = await fetch('/api/model-status');
            const status = await response.json();
            return status;
        } catch (error) {
            console.error('Error checking model status:', error);
            return null;
        }
    }

    async function pollModelLoading() {
        const overlay = document.getElementById('model-loading-overlay');
        const progressBar = document.getElementById('model-progress-bar');
        const stageText = document.getElementById('model-loading-stage');

        while (true) {
            const status = await checkModelStatus();
            
            if (status) {
                progressBar.style.width = `${status.progress}%`;
                stageText.textContent = status.stage;

                if (status.error) {
                    stageText.textContent = `Error: ${status.error}`;
                    stageText.style.color = '#ef4444';
                    break;
                }

                if (status.is_loaded) {
                    overlay.classList.add('hidden');
                    optimizeBtn.disabled = false;
                    console.log('Model loaded successfully!');
                    break;
                }
            }

            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }

    // Start polling for model status
    pollModelLoading();

    // ==================== OPTIMIZATION ====================

    async function optimizeCurrentLayout() {
        const layoutData = captureCurrentLayout();
        
        if (!layoutData || layoutData.furniture.length === 0) {
            alert('Please add some furniture to your layout before optimizing.');
            return;
        }

        try {
            // Disable optimize button
            optimizeBtn.disabled = true;
            optimizeBtn.textContent = '⏳ Optimizing...';

            // Show optimization status
            const statusToast = document.getElementById('optimization-status');
            statusToast.classList.remove('hidden');

            // Submit optimization request
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ layout: layoutData })
            });

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Optimization failed');
            }

            currentJobId = result.job_id;
            console.log('Optimization submitted:', result);

            // Poll for optimization status
            pollOptimizationStatus(currentJobId);

        } catch (error) {
            console.error('Optimization error:', error);
            alert(`Optimization failed: ${error.message}`);
            optimizeBtn.disabled = false;
            optimizeBtn.textContent = 'Optimize Layout';
            document.getElementById('optimization-status').classList.add('hidden');
        }
    }

    async function pollOptimizationStatus(jobId) {
        const statusToast = document.getElementById('optimization-status');
        const statusTitle = document.getElementById('opt-status-title');
        const statusMessage = document.getElementById('opt-status-message');
        const progressBar = document.getElementById('opt-progress-bar');

        while (true) {
            try {
                const response = await fetch(`/api/optimization-status/${jobId}`);
                const status = await response.json();

                progressBar.style.width = `${status.progress || 0}%`;

                if (status.status === 'queued') {
                    statusMessage.textContent = 'Waiting in queue...';
                } else if (status.status === 'processing') {
                    statusMessage.textContent = 'AI is optimizing your layout...';
                } else if (status.status === 'completed') {
                    statusTitle.textContent = '✅ Optimization Complete!';
                    statusMessage.textContent = `Found ${status.violations_count || 0} violations`;
                    progressBar.style.width = '100%';
                    
                    // Store optimized layout
                    optimizedLayoutData = status.output_layout;
                    
                    // Automatically show optimized layout
                    showOptimizedLayout();
                    
                    // Re-enable optimize button
                    optimizeBtn.disabled = false;
                    optimizeBtn.textContent = 'Optimize Layout';

                    // Hide status after 3 seconds
                    setTimeout(() => {
                        statusToast.classList.add('hidden');
                    }, 3000);
                    
                    break;
                } else if (status.status === 'failed') {
                    statusTitle.textContent = '❌ Optimization Failed';
                    statusMessage.textContent = status.error || 'Unknown error';
                    statusTitle.style.color = '#ef4444';
                    
                    optimizeBtn.disabled = false;
                    optimizeBtn.textContent = 'Optimize Layout';

                    setTimeout(() => {
                        statusToast.classList.add('hidden');
                    }, 5000);
                    
                    break;
                }

                await new Promise(resolve => setTimeout(resolve, 1000));

            } catch (error) {
                console.error('Error polling optimization status:', error);
                break;
            }
        }
    }

    function captureCurrentLayout() {
        if (!roomContainer) return null;

        const layoutData = {
            room: {
                width: roomContainer.offsetWidth,
                height: roomContainer.offsetHeight
            },
            furniture: [],
            openings: []
        };

        roomContainer.querySelectorAll('.furniture').forEach(obj => {
            layoutData.furniture.push({
                name: obj.dataset.name,
                x: obj.offsetLeft,
                y: obj.offsetTop,
                width: obj.offsetWidth,
                height: obj.offsetHeight,
                zHeight: parseInt(obj.dataset.zHeight),
                rotation: getRotationAngle(obj)
            });
        });

        roomContainer.querySelectorAll('.wall-feature').forEach(obj => {
            const isVertical = obj.classList.contains('on-vertical-wall');
            const opening = {
                type: obj.dataset.name,
                wall: obj.dataset.wall,
                position: isVertical ? obj.offsetTop : obj.offsetLeft,
                size: isVertical ? obj.offsetHeight : obj.offsetWidth,
                openingHeight: parseInt(obj.dataset.openingHeight)
            };
            if (opening.type === 'window') {
                opening.heightFromGround = parseInt(obj.dataset.heightFromGround);
            }
            layoutData.openings.push(opening);
        });

        return layoutData;
    }

    function showOptimizedLayout() {
        if (!optimizedLayoutData) {
            alert('No optimized layout available.');
            return;
        }

        // Show the optimized canvas section
        document.getElementById('optimized-canvas-section').style.display = 'flex';
        
        // Update canvas container layout
        const canvasContainer = document.getElementById('canvas-container');
        canvasContainer.classList.remove('single-view');
        canvasContainer.classList.add('side-by-side');

        // Show view toggle
        viewToggleContainer.style.display = 'flex';

        // Build optimized layout
        buildLayoutFromJSON(optimizedLayoutData, optimizedCanvas, true);
    }

    // View mode switching
    viewSideBySideBtn.addEventListener('click', () => {
        currentViewMode = 'side-by-side';
        const canvasContainer = document.getElementById('canvas-container');
        canvasContainer.classList.add('side-by-side');
        viewSideBySideBtn.classList.add('active');
    });
    
    optimizeBtn.addEventListener('click', optimizeCurrentLayout);

    // ==================== SIDEBAR TOGGLE ====================
    
    toggleSidebarBtn.addEventListener('click', () => {
        floatingSidebar.classList.toggle('hidden');
    });

    closeSidebarBtn.addEventListener('click', () => {
        floatingSidebar.classList.add('hidden');
    });

    // ==================== FURNITURE BOUNDS ====================

    async function precomputeBounds() {
        const promises = furnitureCatalog.map(item => new Promise(resolve => {
            const img = new Image();
            img.onload = () => {
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = img.naturalWidth;
                tempCanvas.height = img.naturalHeight;
                tempCtx.drawImage(img, 0, 0);
                const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                const data = imageData.data;
                let minX = tempCanvas.width, minY = tempCanvas.height, maxX = 0, maxY = 0;
                for (let y = 0; y < tempCanvas.height; y++) {
                    for (let x = 0; x < tempCanvas.width; x++) {
                        const i = (y * tempCanvas.width + x) * 4;
                        if (data[i + 3] > 0) {
                            minX = Math.min(minX, x);
                            minY = Math.min(minY, y);
                            maxX = Math.max(maxX, x);
                            maxY = Math.max(maxY, y);
                        }
                    }
                }
                const boundsWidth = maxX - minX + 1 || tempCanvas.width;
                const boundsHeight = maxY - minY + 1 || tempCanvas.height;
                furnitureBounds[item.name] = { 
                    minX, 
                    minY, 
                    width: boundsWidth, 
                    height: boundsHeight, 
                    naturalScale: item.width / boundsWidth 
                };
                furnitureRatios[item.name] = img.naturalHeight / img.naturalWidth;
                resolve();
            };
            img.onerror = () => {
                furnitureBounds[item.name] = { 
                    minX: 0, 
                    minY: 0, 
                    width: item.width, 
                    height: item.height, 
                    naturalScale: 1 
                };
                furnitureRatios[item.name] = item.height / item.width;
                resolve();
            };
            img.src = item.image;
        }));
        await Promise.all(promises);
    }

    function getRotationAngle(obj) {
        const match = obj.style.transform && obj.style.transform.match(/rotate\(([^deg]+)deg\)/);
        return match ? parseFloat(match[1]) : 0;
    }

    function updateFurnitureDimensionLabel(obj) {
        const dimDisplay = obj.querySelector('.dimension-display');
        if (!dimDisplay) return;
        const angle = getRotationAngle(obj);
        dimDisplay.textContent = `${Math.round(obj.offsetWidth)}x${Math.round(obj.offsetHeight)}x${obj.dataset.zHeight} cm (${Math.round(angle)}°)`;
    }

    function getVertices(obj) {
        const box = { x: obj.offsetLeft, y: obj.offsetTop, w: obj.offsetWidth, h: obj.offsetHeight };
        const angle = getRotationAngle(obj) * Math.PI / 180;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const cx = box.x + box.w / 2;
        const cy = box.y + box.h / 2;
        const vertices = [];
        for (let i = 0; i < 4; i++) {
            const w_half = (i < 2 ? box.w : -box.w) / 2;
            const h_half = (i % 3 === 0 ? box.h : -box.h) / 2;
            vertices.push({
                x: cx + (w_half * cos - h_half * sin),
                y: cy + (w_half * sin + h_half * cos)
            });
        }
        return vertices;
    }

    function checkCollision(obj1, obj2) {
        const v1 = getVertices(obj1);
        const v2 = getVertices(obj2);
        const axes = [
            v1[0].x - v1[1].x, v1[0].y - v1[1].y, 
            v1[1].x - v1[2].x, v1[1].y - v1[2].y, 
            v2[0].x - v2[1].x, v2[0].y - v2[1].y, 
            v2[1].x - v2[2].x, v2[1].y - v2[2].y
        ].map((val, i, arr) => i % 2 === 0 ? { x: -arr[i + 1], y: val } : null).filter(v => v);
        
        for (let axis of axes) {
            let p1 = v1.map(v => v.x * axis.x + v.y * axis.y);
            let p2 = v2.map(v => v.x * axis.x + v.y * axis.y);
            let min1 = Math.min(...p1), max1 = Math.max(...p1);
            let min2 = Math.min(...p2), max2 = Math.max(...p2);
            if (max1 < min2 || max2 < min1) return false;
        }
        return true;
    }

    function isOverlapping(movingObj, potentialState, containerElement) {
        const clone = document.createElement('div');
        clone.style.position = 'absolute';
        clone.style.left = `${potentialState.x}px`;
        clone.style.top = `${potentialState.y}px`;
        clone.style.width = `${potentialState.width}px`;
        clone.style.height = `${potentialState.height}px`;
        clone.style.transform = `rotate(${potentialState.rotation}deg)`;
        const otherObjects = [...containerElement.querySelectorAll('.furniture')].filter(
            child => child.id !== (movingObj && movingObj.id)
        );
        for (let other of otherObjects) {
            if (checkCollision(clone, other)) return true;
        }
        return false;
    }

    function populateSidebar() {
        [furnitureList, architecturalList].forEach(list => list.innerHTML = '');
        
        const createSidebarItem = (item) => {
            const div = document.createElement('div');
            div.className = 'bg-slate-800 hover:bg-slate-700 p-3 rounded-lg cursor-grab flex flex-col items-center shadow-sm border border-slate-700 transition hover:shadow-md hover:scale-105';
            div.draggable = true;
            div.dataset.item = JSON.stringify(item);
            
            const bounds = furnitureBounds[item.name] || { width: item.width, height: item.height };
            const aspectRatio = bounds.height / bounds.width;
            const baseWidth = 60;
            const scaledHeight = Math.max(30, baseWidth * aspectRatio);
            
            div.innerHTML = `
                <div class="sidebar-icon" style="width:${baseWidth + 16}px;height:${scaledHeight + 16}px;">
                    <img src="${item.image}" 
                         style="width:${baseWidth}px;height:${scaledHeight}px;object-fit:contain;" 
                         draggable="false" />
                </div>
                <span class="font-medium text-xs text-slate-200 mt-2">${item.name}</span>
            `;
            
            div.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', e.currentTarget.dataset.item);
            });
            
            return div;
        };
        
        furnitureCatalog.forEach(item => furnitureList.appendChild(createSidebarItem(item)));
        architecturalCatalog.forEach(item => architecturalList.appendChild(createSidebarItem(item)));
    }

    function addFurnitureEventListeners(obj, containerElement) {
        const rotateHandle = obj.querySelector('.rotate-handle');
        const resizeHandle = obj.querySelector('.resize-handle');

        obj.addEventListener('mousedown', e => {
            if (e.target.classList.contains('furniture') || e.target.closest('.furniture')) {
                if (!e.target.classList.contains('resize-handle') && !e.target.classList.contains('rotate-handle')) {
                    selectObject(obj);
                    let isDragging = true;
                    const startMouseX = e.clientX;
                    const startMouseY = e.clientY;
                    const startCenterX = obj.offsetLeft + obj.offsetWidth / 2;
                    const startCenterY = obj.offsetTop + obj.offsetHeight / 2;
                    document.body.style.cursor = 'move';
                    e.stopPropagation();
                    
                    function dragMove(e) {
                        if (!isDragging) return;
                        let newCenterX = startCenterX + (e.clientX - startMouseX);
                        let newCenterY = startCenterY + (e.clientY - startMouseY);
                        const angleRad = getRotationAngle(obj) * Math.PI / 180;
                        const w = obj.offsetWidth;
                        const h = obj.offsetHeight;
                        const rotatedHalfWidth = (Math.abs(Math.cos(angleRad)) * w + Math.abs(Math.sin(angleRad)) * h) / 2;
                        const rotatedHalfHeight = (Math.abs(Math.sin(angleRad)) * w + Math.abs(Math.cos(angleRad)) * h) / 2;
                        newCenterX = Math.max(rotatedHalfWidth, Math.min(newCenterX, containerElement.offsetWidth - rotatedHalfWidth));
                        newCenterY = Math.max(rotatedHalfHeight, Math.min(newCenterY, containerElement.offsetHeight - rotatedHalfHeight));
                        const potentialState = { 
                            x: newCenterX - w / 2, 
                            y: newCenterY - h / 2, 
                            width: w, 
                            height: h, 
                            rotation: getRotationAngle(obj) 
                        };
                        if (!isOverlapping(obj, potentialState, containerElement)) {
                            obj.style.left = `${potentialState.x}px`;
                            obj.style.top = `${potentialState.y}px`;
                            obj.classList.remove('colliding');
                        } else {
                            obj.classList.add('colliding');
                        }
                    }
                    
                    function dragEnd() {
                        isDragging = false;
                        obj.classList.remove('colliding');
                        document.body.style.cursor = 'default';
                        document.removeEventListener('mousemove', dragMove);
                        document.removeEventListener('mouseup', dragEnd);
                    }
                    
                    document.addEventListener('mousemove', dragMove);
                    document.addEventListener('mouseup', dragEnd);
                }
            }
        });

        resizeHandle.addEventListener('mousedown', e => {
            e.stopPropagation();
            selectObject(obj);
            let isResizing = true;
            const startWidth = obj.offsetWidth;
            const startHeight = obj.offsetHeight;
            const startMouseX = e.clientX;
            const startMouseY = e.clientY;
            
            function resizeMove(e) {
                if (!isResizing) return;
                const newWidth = startWidth + (e.clientX - startMouseX);
                const newHeight = startHeight + (e.clientY - startMouseY);
                const potentialState = { 
                    x: obj.offsetLeft, 
                    y: obj.offsetTop, 
                    width: newWidth, 
                    height: newHeight, 
                    rotation: getRotationAngle(obj) 
                };
                if (!isOverlapping(obj, potentialState, containerElement)) {
                    obj.style.width = `${Math.max(20, newWidth)}px`;
                    obj.style.height = `${Math.max(20, newHeight)}px`;
                    const img = obj.querySelector('img');
                    const scaleX = newWidth / startWidth;
                    const scaleY = newHeight / startHeight;
                    img.style.transformOrigin = 'center center';
                    img.style.transform = `translate(-50%, -50%) scale(${scaleX}, ${scaleY})`;
                    img.style.position = 'absolute';
                    img.style.left = '50%';
                    img.style.top = '50%';

                    updateFurnitureDimensionLabel(obj);
                    obj.classList.remove('colliding');
                } else {
                    obj.classList.add('colliding');
                }
            }
            
            function resizeEnd() {
                isResizing = false;
                obj.classList.remove('colliding');
                document.removeEventListener('mousemove', resizeMove);
                document.removeEventListener('mouseup', resizeEnd);
            }
            
            document.addEventListener('mousemove', resizeMove);
            document.addEventListener('mouseup', resizeEnd);
        });

        rotateHandle.addEventListener('mousedown', e => {
            e.stopPropagation();
            selectObject(obj);
            let isRotating = true;
            const startAngle = getRotationAngle(obj);
            const startMouseX = e.clientX;
            const startMouseY = e.clientY;
            const rect = obj.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            const startMouseAngle = Math.atan2(startMouseY - centerY, startMouseX - centerX) * (180 / Math.PI);
            
            function rotateMove(e) {
                if (!isRotating) return;
                const currentMouseX = e.clientX;
                const currentMouseY = e.clientY;
                const currentMouseAngle = Math.atan2(currentMouseY - centerY, currentMouseX - centerX) * (180 / Math.PI);
                const rawRotation = startAngle + (currentMouseAngle - startMouseAngle);
                let finalRotation = rawRotation;
                let isSnapped = false;
                
                for (let angle of [0, 90, 180, 270, 360, -90, -180, -270]) {
                    if (Math.abs(rawRotation - angle) < 10) {
                        finalRotation = angle % 360;
                        isSnapped = true;
                        break;
                    }
                }
                
                const potentialState = { 
                    x: obj.offsetLeft, 
                    y: obj.offsetTop, 
                    width: obj.offsetWidth, 
                    height: obj.offsetHeight, 
                    rotation: finalRotation 
                };
                
                if (!isOverlapping(obj, potentialState, containerElement)) {
                    rotateHandle.classList.toggle('snapped', isSnapped);
                    obj.style.transform = `rotate(${finalRotation}deg)`;
                    updateFurnitureDimensionLabel(obj);
                    obj.classList.remove('colliding');
                } else {
                    obj.classList.add('colliding');
                }
            }
            
            function rotateEnd() {
                isRotating = false;
                rotateHandle.classList.remove('snapped');
                obj.classList.remove('colliding');
                document.removeEventListener('mousemove', rotateMove);
                document.removeEventListener('mouseup', rotateEnd);
            }
            
            document.addEventListener('mousemove', rotateMove);
            document.addEventListener('mouseup', rotateEnd);
        });
    }

    function addWallFeatureEventListeners(obj, containerElement) {
        let action = null;
        let startMousePos, startObjPos, startSize;
        const isVertical = obj.classList.contains('on-vertical-wall');
        
        obj.addEventListener('mousemove', e => {
            if (action) return;
            const rect = obj.getBoundingClientRect();
            const clickPos = isVertical ? e.clientY : e.clientX;
            const startEdge = isVertical ? rect.top : rect.left;
            const endEdge = isVertical ? rect.bottom : rect.right;
            obj.style.cursor = (Math.abs(clickPos - startEdge) < 10 || Math.abs(clickPos - endEdge) < 10) 
                ? (isVertical ? 'ns-resize' : 'ew-resize') 
                : 'move';
        });
        
        obj.addEventListener('mousedown', e => {
            e.stopPropagation();
            selectObject(obj);
            const rect = obj.getBoundingClientRect();
            const clickPos = isVertical ? e.clientY : e.clientX;
            const startEdge = isVertical ? rect.top : rect.left;
            const endEdge = isVertical ? rect.bottom : rect.right;
            
            if (Math.abs(clickPos - startEdge) < 10) action = 'resize-start';
            else if (Math.abs(clickPos - endEdge) < 10) action = 'resize-end';
            else action = 'drag';
            
            startMousePos = isVertical ? e.clientY : e.clientX;
            startObjPos = isVertical ? obj.offsetTop : obj.offsetLeft;
            startSize = isVertical ? obj.offsetHeight : obj.offsetWidth;
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
        
        function onMouseMove(e) {
            if (!action) return;
            e.preventDefault();
            const currentMousePos = isVertical ? e.clientY : e.clientX;
            const delta = currentMousePos - startMousePos;
            
            if (action === 'drag') {
                if (isVertical) {
                    let newTop = startObjPos + delta;
                    newTop = Math.max(0, Math.min(newTop, containerElement.offsetHeight - obj.offsetHeight));
                    obj.style.top = `${newTop}px`;
                } else {
                    let newLeft = startObjPos + delta;
                    newLeft = Math.max(0, Math.min(newLeft, containerElement.offsetWidth - obj.offsetWidth));
                    obj.style.left = `${newLeft}px`;
                }
            } else {
                if (isVertical) {
                    if (action === 'resize-start') {
                        const newHeight = startSize - delta;
                        if (newHeight > 30) {
                            obj.style.height = `${newHeight}px`;
                            obj.style.top = `${startObjPos + delta}px`;
                        }
                    } else {
                        const newHeight = startSize + delta;
                        if (newHeight > 30) obj.style.height = `${newHeight}px`;
                    }
                } else {
                    if (action === 'resize-start') {
                        const newWidth = startSize - delta;
                        if (newWidth > 30) {
                            obj.style.width = `${newWidth}px`;
                            obj.style.left = `${startObjPos + delta}px`;
                        }
                    } else {
                        const newWidth = startSize + delta;
                        if (newWidth > 30) obj.style.width = `${newWidth}px`;
                    }
                }
                updateWallFeatureDimensionLabel(obj);
            }
        }
        
        function onMouseUp() {
            action = null;
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        }
    }

    function updateWallFeatureDimensionLabel(obj) {
        const dimDisplay = obj.querySelector('.dimension-display');
        if (!dimDisplay) return;
        const width = obj.classList.contains('on-vertical-wall') ? obj.offsetHeight : obj.offsetWidth;
        if (obj.classList.contains('window')) {
            dimDisplay.textContent = `W:${width}, H:${obj.dataset.openingHeight}, Sill:${obj.dataset.heightFromGround} cm`;
        } else {
            dimDisplay.textContent = `W:${width}, H:${obj.dataset.openingHeight} cm`;
        }
    }

    function updateRoomDimensions(containerElement) {
        const display = containerElement.querySelector('.room-dimension');
        if (display) display.textContent = `${containerElement.offsetWidth} x ${containerElement.offsetHeight} cm`;
    }

    function updateWallFeaturesPosition(containerElement) {
        containerElement.querySelectorAll('.wall-feature').forEach(obj => {
            const wall = obj.dataset.wall;
            const isVertical = obj.classList.contains('on-vertical-wall');
            
            if (wall === 'bottom') obj.style.top = `${containerElement.offsetHeight - 5}px`;
            else if (wall === 'right') obj.style.left = `${containerElement.offsetWidth - 5}px`;
            
            if (!isVertical) {
                const maxLeft = containerElement.offsetWidth - obj.offsetWidth;
                if (obj.offsetLeft > maxLeft) obj.style.left = `${maxLeft}px`;
            } else {
                const maxTop = containerElement.offsetHeight - obj.offsetHeight;
                if (obj.offsetTop > maxTop) obj.style.top = `${maxTop}px`;
            }
        });
    }

    function selectObject(obj) {
        if (selectedObject) selectedObject.classList.remove('selected');
        selectedObject = obj;
        contextualSettings.innerHTML = '';
        
        if (!obj) {
            deleteBtn.disabled = true;
            return;
        }
        
        selectedObject.classList.add('selected');
        deleteBtn.disabled = false;
        
        if (obj.classList.contains('furniture')) {
            contextualSettings.innerHTML = `
                <label for="cs-height" class="text-sm text-slate-200 mr-2">Height(cm):</label>
                <input type="number" id="cs-height" 
                       class="w-20 border-gray-300 rounded-md shadow-sm text-sm bg-slate-700 text-white border-slate-600 px-2 py-1" 
                       value="${obj.dataset.zHeight}">
                <button id="cs-set-height" 
                        class="px-3 py-1 bg-blue-500 text-white text-xs rounded ml-2 hover:bg-blue-600">Set</button>
            `;
            document.getElementById('cs-set-height').addEventListener('click', () => {
                obj.dataset.zHeight = document.getElementById('cs-height').value;
                updateFurnitureDimensionLabel(obj);
            });
        } else if (obj.classList.contains('window')) {
            contextualSettings.innerHTML = `
                <label for="cs-sill" class="text-sm text-slate-200 mr-2">Sill(cm):</label>
                <input type="number" id="cs-sill" 
                       class="w-20 border-gray-300 rounded-md text-sm bg-slate-700 text-white border-slate-600 px-2 py-1" 
                       value="${obj.dataset.heightFromGround}">
                <label for="cs-o-height" class="text-sm text-slate-200 ml-2 mr-2">Height(cm):</label>
                <input type="number" id="cs-o-height" 
                       class="w-20 border-gray-300 rounded-md text-sm bg-slate-700 text-white border-slate-600 px-2 py-1" 
                       value="${obj.dataset.openingHeight}">
            `;
            document.getElementById('cs-sill').addEventListener('change', e => {
                obj.dataset.heightFromGround = e.target.value;
                updateWallFeatureDimensionLabel(obj);
            });
            document.getElementById('cs-o-height').addEventListener('change', e => {
                obj.dataset.openingHeight = e.target.value;
                updateWallFeatureDimensionLabel(obj);
            });
        } else if (obj.classList.contains('door')) {
            contextualSettings.innerHTML = `
                <label for="cs-o-height" class="text-sm text-slate-200 mr-2">Height(cm):</label>
                <input type="number" id="cs-o-height" 
                       class="w-20 border-gray-300 rounded-md text-sm bg-slate-700 text-white border-slate-600 px-2 py-1" 
                       value="${obj.dataset.openingHeight}">
                <button id="cs-set-o-height" 
                        class="px-3 py-1 bg-blue-500 text-white text-xs rounded ml-2 hover:bg-blue-600">Set</button>
            `;
            document.getElementById('cs-set-o-height').addEventListener('click', () => {
                obj.dataset.openingHeight = document.getElementById('cs-o-height').value;
                updateWallFeatureDimensionLabel(obj);
            });
        }
    }

    const deselectAll = () => selectObject(null);

    function findClosestWall(x, y, roomWidth, roomHeight) {
        const dists = { 
            top: y, 
            bottom: roomHeight - y, 
            left: x, 
            right: roomWidth - x 
        };
        const closest = Object.keys(dists).reduce((a, b) => dists[a] < dists[b] ? a : b);
        let position = (closest === 'top' || closest === 'bottom') ? x : y;
        return { name: closest, position: position };
    }

    function createFurnitureObject(itemData, pos, loaded = false, containerElement) {
        objectCounter++;
        const obj = document.createElement('div');
        obj.id = `object-${objectCounter}`;
        obj.className = 'furniture';
        obj.dataset.name = itemData.name;
        obj.dataset.zHeight = itemData.zHeight;
        obj.style.left = `${pos.x}px`;
        obj.style.top = `${pos.y}px`;
        obj.style.transform = `rotate(${pos.rotation || 0}deg)`;

        // Append handles and dim display first
        obj.innerHTML = `
            <div class="handle resize-handle"></div>
            <div class="handle rotate-handle"></div>
            <div class="dimension-display"></div>
        `;

        const img = document.createElement('img');
        img.src = itemData.image;
        img.alt = itemData.name;
        img.style.position = 'absolute';
        img.style.top = '0';
        img.style.left = '0';
        img.draggable = false;
        obj.appendChild(img);

        // Use precomputed bounds to set size and position
        const bounds = furnitureBounds[itemData.name];
        if (bounds) {
            const scale = (pos.width || itemData.width) / bounds.width;
            obj.style.width = `${pos.width || itemData.width}px`;
            obj.style.height = `${pos.height || (bounds.height * scale)}px`;
            // Position img to crop to tight bounds
            img.style.width = `${img.naturalWidth * scale}px`;
            img.style.height = `${img.naturalHeight * scale}px`;
            img.style.left = `${-bounds.minX * scale}px`;
            img.style.top = `${-bounds.minY * scale}px`;
        } else {
            // Fallback
            obj.style.width = `${pos.width || itemData.width}px`;
            obj.style.height = `${pos.height || itemData.height}px`;
        }

        img.onload = () => {
            if (!bounds) {
                // Recompute if not precomputed
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = img.naturalWidth;
                tempCanvas.height = img.naturalHeight;
                tempCtx.drawImage(img, 0, 0);
                const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                const data = imageData.data;
                let minX = tempCanvas.width, minY = tempCanvas.height, maxX = 0, maxY = 0;
                for (let y = 0; y < tempCanvas.height; y++) {
                    for (let x = 0; x < tempCanvas.width; x++) {
                        const i = (y * tempCanvas.width + x) * 4;
                        if (data[i + 3] > 0) {
                            minX = Math.min(minX, x);
                            minY = Math.min(minY, y);
                            maxX = Math.max(maxX, x);
                            maxY = Math.max(maxY, y);
                        }
                    }
                }
                const bWidth = maxX - minX + 1;
                const bHeight = maxY - minY + 1;
                const scale = (pos.width || itemData.width) / bWidth;
                obj.style.width = `${pos.width || itemData.width}px`;
                obj.style.height = `${bHeight * scale}px`;
                img.style.width = `${img.naturalWidth * scale}px`;
                img.style.height = `${img.naturalHeight * scale}px`;
                img.style.left = `${-minX * scale}px`;
                img.style.top = `${-minY * scale}px`;
                furnitureBounds[itemData.name] = { minX, minY, width: bWidth, height: bHeight, naturalScale: scale };
            }
            updateFurnitureDimensionLabel(obj);
        };

        containerElement.appendChild(obj);
        addFurnitureEventListeners(obj, containerElement);
        
        if (!loaded) selectObject(obj);
    }

    function createWallFeature(type, options = {}, containerElement) {
        objectCounter++;
        const feature = document.createElement('div');
        feature.id = `object-${objectCounter}`;
        feature.className = `wall-feature ${type}`;
        feature.dataset.name = type;

        const wall = options.wall || 'top';
        feature.dataset.wall = wall;

        feature.dataset.openingHeight = options.openingHeight || (type === 'door' ? 210 : 100);
        if (type === 'window') feature.dataset.heightFromGround = options.heightFromGround || 90;

        feature.innerHTML = `<div class="dimension-display"></div>`;

        if (type === 'door') {
            feature.style.background = 'linear-gradient(to right, #ff5100ff 0%, #ff6600ff 100%)';
            feature.style.border = '2px solid #ff782fff';
            feature.style.borderRadius = '3px';
            feature.style.boxShadow = '0 0 8px rgba(255, 115, 0, 0.5)';
        } else if (type === 'window') {
            feature.style.background = 'linear-gradient(145deg, #38bdf8 0%, #0ea5e9 100%)';
            feature.style.border = '2px solid #7dd3fc';
            feature.style.borderRadius = '3px';
            feature.style.boxShadow = '0 0 8px rgba(56,189,248,0.5)';
        }

        const wallThickness = 10;
        if (wall === 'top' || wall === 'bottom') {
            feature.style.width = `${options.size || (type === 'door' ? 90 : 120)}px`;
            feature.style.height = `${wallThickness}px`;
            const position = Math.max(0, Math.min(
                options.position || 100, 
                containerElement.offsetWidth - (options.size || 120)
            ));
            feature.style.left = `${position}px`;
            feature.style.top = wall === 'top' 
                ? `-${wallThickness / 2}px` 
                : `${containerElement.offsetHeight - wallThickness / 2}px`;
        } else {
            feature.classList.add('on-vertical-wall');
            feature.style.height = `${options.size || (type === 'door' ? 90 : 120)}px`;
            feature.style.width = `${wallThickness}px`;
            const position = Math.max(0, Math.min(
                options.position || 100, 
                containerElement.offsetHeight - (options.size || 120)
            ));
            feature.style.top = `${position}px`;
            feature.style.left = wall === 'left' 
                ? `-${wallThickness / 2}px` 
                : `${containerElement.offsetWidth - wallThickness / 2}px`;
        }

        containerElement.appendChild(feature);
        addWallFeatureEventListeners(feature, containerElement);
        updateWallFeatureDimensionLabel(feature);
        selectObject(feature);
    }

    function handleDrop(e, containerElement) {
        e.preventDefault();
        const itemData = JSON.parse(e.dataTransfer.getData('text/plain'));
        const roomRect = containerElement.getBoundingClientRect();
        const x = e.clientX - roomRect.left;
        const y = e.clientY - roomRect.top;
        
        if (itemData.type === 'door' || itemData.type === 'window') {
            const closestWall = findClosestWall(x, y, containerElement.offsetWidth, containerElement.offsetHeight);
            createWallFeature(itemData.type, { 
                wall: closestWall.name, 
                position: closestWall.position - (itemData.width / 2) 
            }, containerElement);
        } else {
            // Calculate adjusted height based on bounds
            const bounds = furnitureBounds[itemData.name];
            const adjHeight = bounds ? (itemData.width / bounds.width) * bounds.height : itemData.height;

            const pos = { 
                x: x - (itemData.width / 2), 
                y: y - (adjHeight / 2),  // ✅ Use adjHeight
                rotation: 0,
                width: itemData.width,
                height: adjHeight  // ✅ Use adjHeight
            };
            const checkState = { 
                x: pos.x, 
                y: pos.y, 
                width: itemData.width, 
                height: adjHeight, 
                rotation: 0 
            };
            if (!isOverlapping(null, checkState, containerElement)) {
                createFurnitureObject(itemData, pos, false, containerElement);
            } else {
                console.warn("Cannot place object here: Overlap detected.");
            }
        }
    }

    function addRoomResizeListener(handle, position, containerElement) {
        handle.addEventListener('mousedown', e => {
            e.stopPropagation();
            let isResizing = true;
            const startX = e.clientX;
            const startY = e.clientY;
            const startWidth = containerElement.offsetWidth;
            const startHeight = containerElement.offsetHeight;
            document.body.style.cursor = handle.style.cursor;
            
            function resizeMove(e) {
                if (!isResizing) return;
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                let newWidth = startWidth;
                let newHeight = startHeight;
                
                if (position.includes('right')) newWidth += dx;
                if (position.includes('left')) newWidth -= dx;
                if (position.includes('bottom')) newHeight += dy;
                if (position.includes('top')) newHeight -= dy;
                
                containerElement.style.width = `${Math.max(200, newWidth)}px`;
                containerElement.style.height = `${Math.max(200, newHeight)}px`;
                updateRoomDimensions(containerElement);
                updateWallFeaturesPosition(containerElement);
            }
            
            function resizeEnd() {
                isResizing = false;
                document.body.style.cursor = 'default';
                document.removeEventListener('mousemove', resizeMove);
                document.removeEventListener('mouseup', resizeEnd);
            }
            
            document.addEventListener('mousemove', resizeMove);
            document.addEventListener('mouseup', resizeEnd);
        });
    }

    function saveLayout() {
        const layoutData = captureCurrentLayout();
        
        const blob = new Blob([JSON.stringify(layoutData, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'room-layout.json';
        a.click();
        URL.revokeObjectURL(a.href);
    }

    function buildLayoutFromJSON(data, targetCanvas, isOptimized = false) {
        deselectAll();
        targetCanvas.innerHTML = '';
        initializeApp(data.room.width, data.room.height, targetCanvas, isOptimized);
        
        const container = isOptimized ? optimizedRoomContainer : roomContainer;
        
        (data.openings || []).forEach(o => createWallFeature(o.type, o, container));
        
        (data.furniture || []).forEach(f => {
            const catalogItem = furnitureCatalog.find(item => item.name === f.name);
            if (catalogItem) {
                const itemData = { ...catalogItem };
                const pos = { 
                    x: f.x, 
                    y: f.y, 
                    rotation: f.rotation,
                    width: f.width,
                    height: f.height
                };
                itemData.zHeight = f.zHeight;
                createFurnitureObject(itemData, pos, true, container);
            }
        });
        
        deselectAll();
    }

    function initializeApp(width = 800, height = 600, targetCanvas = canvas, isOptimized = false) {
        targetCanvas.innerHTML = '';
        const container = document.createElement('div');
        container.id = isOptimized ? 'optimized-room-container' : 'room-container';
        container.style.width = `${width}px`;
        container.style.height = `${height}px`;
        container.style.backgroundImage = "linear-gradient(to right, #334155 1px, transparent 1px), linear-gradient(to bottom, #334155 1px, transparent 1px)";
        container.style.backgroundSize = "40px 40px";
        targetCanvas.appendChild(container);
        
        if (isOptimized) {
            optimizedRoomContainer = container;
        } else {
            roomContainer = container;
        }
        
        const roomDimDisplay = document.createElement('div');
        roomDimDisplay.className = 'dimension-display room-dimension';
        container.appendChild(roomDimDisplay);
        updateRoomDimensions(container);
        
        ['top-left', 'top-right', 'bottom-left', 'bottom-right'].forEach(pos => {
            const handle = document.createElement('div');
            handle.className = `handle room-handle ${pos}`;
            container.appendChild(handle);
            addRoomResizeListener(handle, pos, container);
        });
        
        container.addEventListener('dragover', e => e.preventDefault());
        container.addEventListener('drop', e => handleDrop(e, container));
    }

    deleteBtn.addEventListener('click', () => {
        if (selectedObject) {
            selectedObject.remove();
            deselectAll();
        }
    });

    canvas.addEventListener('click', e => {
        if (e.target.id === 'canvas' || e.target.id === 'room-container') deselectAll();
    });

    loadBtn.addEventListener('click', () => loadLayoutInput.click());
    
    loadLayoutInput.addEventListener('change', e => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (re) => {
                try {
                    buildLayoutFromJSON(JSON.parse(re.target.result), canvas, false);
                } catch (err) {
                    alert('Error: Could not load layout from file.');
                    console.error(err);
                }
            };
            reader.readAsText(file);
            e.target.value = null;
        }
    });

    saveBtn.addEventListener('click', saveLayout);
    toggleDimsSwitch.addEventListener('change', () => {
        document.body.classList.toggle('hide-dims', !toggleDimsSwitch.checked);
    });

    await precomputeBounds();
    populateSidebar();
    initializeApp();
});