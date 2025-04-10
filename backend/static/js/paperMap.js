// paperMap.js
class MultiClusterMap {
    constructor(containerId, tooltipId) {
        this.containerId = containerId;
        this.tooltipId = tooltipId;

        // Grab DOM elements
        this.container = document.getElementById(containerId);
        this.tooltip = document.getElementById(tooltipId);

        // Distances for cluster logic
        this.centerThreshold = 120;   // near cluster's center => become new center
        this.detachThreshold = 650;   // far => new cluster

        // Tooltip timing
        this.tooltipDelay = 1000;
        this.hoverTimer = null;

        // Array of clusters, each with its own simulation + group
        this.clusters = [];
        // Give each cluster a numeric ID
        this.clusterCounter = 0;

        // Zoom configuration
        this.zoomExtent = [0.2, 5]; // Min and max zoom levels
        this.currentZoom = 1;       // Track zoom level

        // If you have a loading or "message" state, we can display that
        this.showMessage("Search for research papers to visualize");
    }

    //--------------------------------------
    //  LOADING & STATUS MESSAGES
    //--------------------------------------
    showMessage(msg) {
        this.container.innerHTML = `<div class="map-message">${msg}</div>`;
    }

    showLoading() {
        this.container.innerHTML = '<div class="loading-map">Working on it...</div>';
    }

    showDragHint() {
        const hint = d3.select(`#${this.containerId}`)
            .append("div")
            .attr("class", "hint-message")
            .html('Tip: Drag a node near its center to become new center, or far away to spawn another cluster. Use mouse wheel to zoom in/out.');

        setTimeout(() => {
            hint.transition().duration(1000).style("opacity", 0).remove();
        }, 5000);
    }

    //--------------------------------------
    //  MAIN SEARCH -> BUILD FIRST CLUSTER
    //--------------------------------------
    async fetchData(query) {
        if (!query || !query.trim()) {
            this.showMessage("Search for research papers to visualize");
            return;
        }
        this.showLoading();

        try {
            const resp = await fetch("/map_data?" + new URLSearchParams({ query }).toString());
            if (!resp.ok) throw new Error(`Network error: ${resp.statusText}`);

            const data = await resp.json();
            if (!data.center) {
                this.showMessage("No results found");
                return;
            }

            // Clear old clusters
            this.clusters = [];
            // Clear the container
            this.container.innerHTML = "";

            // Create a <div> to hold the <svg>
            const containerRect = document.createElement("div");
            containerRect.className = "map-container-rect";
            this.container.appendChild(containerRect);

            // Determine size
            this.width = this.container.clientWidth;
            this.height = this.container.clientHeight;

            // Create main SVG
            this.svg = d3.select(containerRect)
                .append("svg")
                .attr("width", "100%")
                .attr("height", "100%")
                .attr("viewBox", `0 0 ${this.width} ${this.height}`)
                .attr("preserveAspectRatio", "xMidYMid meet");

            // Add a group for all clusters that will be transformed during zoom
            this.zoomGroup = this.svg.append("g")
                .attr("class", "zoom-group");
                
            // Initialize zoom behavior
            this.initZoom();

            // Show a helpful drag hint
            this.showDragHint();

            // Create the initial cluster in center
            this.createCluster(data, { x: this.width / 2, y: this.height / 2 });
        } catch (err) {
            console.error("fetchData error:", err);
            this.showMessage(`Error: ${err.message}`);
        }
    }

    //--------------------------------------
    //  INITIALIZE ZOOM BEHAVIOR
    //--------------------------------------
    initZoom() {
        // Create zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent(this.zoomExtent)
            .on("zoom", (event) => {
                // Store current zoom level
                this.currentZoom = event.transform.k;
                
                // Apply transformation to the zoom group
                this.zoomGroup.attr("transform", event.transform);
                
                // Update tooltip position calculation for zoomed view
                this.updateTooltipPositionForZoom();
            });

        // Apply zoom to SVG
        this.svg.call(this.zoom);
        
        // Add zoom controls
        this.addZoomControls();
    }

    updateTooltipPositionForZoom() {
        // This method will be called during zoom events
        // to adjust tooltip position calculation if needed
        // Currently, we're not doing anything special here,
        // but it's a hook for future enhancements
    }

    addZoomControls() {
        // Create zoom controls div - using the CSS class from the stylesheet
        const controlsDiv = d3.select(`#${this.containerId}`)
            .append("div")
            .attr("class", "zoom-controls");
            
        // Zoom in button
        controlsDiv.append("button")
            .attr("class", "zoom-button")
            .html("+")
            .on("click", () => this.zoomBy(1.2)); // Zoom in by 20%
            
        // Zoom out button
        controlsDiv.append("button")
            .attr("class", "zoom-button")
            .html("−")
            .on("click", () => this.zoomBy(0.8)); // Zoom out by 20%
            
        // Reset zoom button
        controlsDiv.append("button")
            .attr("class", "zoom-button")
            .html("⟲")
            .on("click", () => this.resetZoom());
    }

    zoomBy(factor) {
        // Calculate new zoom level, constrained by zoom extent
        const newScale = Math.max(
            this.zoomExtent[0], 
            Math.min(this.zoomExtent[1], this.currentZoom * factor)
        );
        
        // Use D3's zoom.transform to zoom to the new level
        this.svg.transition()
            .duration(300)
            .call(
                this.zoom.transform,
                d3.zoomIdentity.scale(newScale)
            );
    }

    resetZoom() {
        // Reset to original scale and position
        this.svg.transition()
            .duration(500)
            .call(
                this.zoom.transform,
                d3.zoomIdentity
            );
    }

    //--------------------------------------
    //  CREATE A NEW CLUSTER
    //--------------------------------------
    createCluster(mapData, position) {
        // position => (x,y) where cluster will roughly gather
        const cid = ++this.clusterCounter;

        // Build the data arrays for nodes & links
        // Make local copies so we can store a special property __clusterId
        const centerNode = { ...mapData.center, __clusterId: cid, id: 0 };  // center => id=0
        const relatedNodes = mapData.related.map((r, i) => ({
            ...r, __clusterId: cid, id: i + 1
        }));
        const nodes = [centerNode, ...relatedNodes];
        const links = relatedNodes.map(n => ({
            source: 0,    // center is node id=0
            target: n.id, // the related node
            value: n.score
        }));

        // Create a group for this entire cluster
        // Append to zoomGroup instead of directly to SVG
        const clusterGroup = this.zoomGroup.append("g")
            .attr("class", `cluster-group c${cid}`);

        // Make subgroups for links & nodes
        const linkGroup = clusterGroup.append("g").attr("class", "links-container");
        const nodeGroup = clusterGroup.append("g").attr("class", "nodes-container");

        // Link selection
        const linkSel = linkGroup.selectAll("line")
            .data(links)
            .enter()
            .append("line")
            .attr("class", "paper-link")
            .attr("stroke", "#444")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", d => Math.max(1, d.value * 6));

        // Node selection
        const nodeSel = nodeGroup.selectAll("g.paper-node-group")
            .data(nodes)
            .enter()
            .append("g")
            .attr("class", "paper-node-group")
            .call(d3.drag()
                .on("start", (event, d) => this.dragstarted(event, d, cid))
                .on("drag", (event, d) => this.dragged(event, d, cid))
                .on("end", (event, d) => this.dragended(event, d, cid))
            );

        // Circles
        nodeSel.append("circle")
            .attr("class", "paper-node")
            .attr("r", d => this.getNodeRadius(d))
            .attr("fill", d => this.getNodeColor(d))
            .attr("stroke", "#333")
            .attr("stroke-width", 1.5)
            // Delayed tooltip
            .on("mouseover", (event, d) => this.handleNodeMouseOver(event, d))
            .on("mouseout", () => this.handleNodeMouseOut())
            .on("click", (event, d) => this.openPaperLink(d));

        // Titles
        this.addNodeTitles(nodeSel);

        // Force simulation local to this cluster
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(d => 200 * (1 - d.value)))
            .force("charge", d3.forceManyBody().strength(-1200))
            .force("collision", d3.forceCollide().radius(d => this.getNodeRadius(d) + 25))
            // center the cluster around the specified position
            .force("x", d3.forceX(position.x).strength(0.1))
            .force("y", d3.forceY(position.y).strength(0.1));

        simulation.on("tick", () => {
            // Apply inter-cluster node collision force before updating positions
            if (this.clusters.length > 0) {
                this.applyInterClusterNodeCollision();
            }
            
            linkSel
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            nodeSel
                .attr("transform", d => `translate(${d.x},${d.y})`);
                
            // Update cluster bounds after each tick
            this.updateClusterBounds(cid);
            
            // Apply inter-cluster separation force
            if (this.clusters.length > 1) {
                this.applyClusterSeparationForce();
            }
        });

        // Save cluster info
        this.clusters.push({
            id: cid,
            group: clusterGroup,
            linkGroup,
            nodeGroup,
            nodes,
            links,
            simulation,
            position: position, // Store initial position
            bounds: {           // Will be updated during simulation
                minX: position.x - 200,
                maxX: position.x + 200,
                minY: position.y - 200,
                maxY: position.y + 200,
                centerX: position.x,
                centerY: position.y,
                radius: 200     // Initial estimated radius
            }
        });
        
        // Initial bounds calculation
        this.updateClusterBounds(cid);
        
        // Re-establish the inter-cluster forces now that we have a new cluster
        this.setupGlobalForces();
        
        return this.clusters[this.clusters.length - 1]; // Return the created cluster
    }
    
    //--------------------------------------
    //  GLOBAL FORCES FOR ALL CLUSTERS
    //--------------------------------------
    setupGlobalForces() {
        // This method establishes forces that work across all clusters
        // Call this when clusters are added or removed
        
        // We don't need to do anything if we only have 0 or 1 cluster
        if (this.clusters.length <= 1) return;
        
        // Restart all simulations to apply new forces
        this.clusters.forEach(cluster => {
            if (cluster.simulation) {
                cluster.simulation.alpha(0.3).restart();
            }
        });
    }
    
    //--------------------------------------
    //  INTER-CLUSTER NODE COLLISION
    //--------------------------------------
    applyInterClusterNodeCollision() {
        // Stop if we only have one cluster
        if (this.clusters.length <= 1) return;
        
        // Extra padding for nodes from different clusters
        // (should be larger than intra-cluster collision padding)
        const INTER_CLUSTER_PADDING = 10;
        
        // Quadratic complexity, but for small numbers of clusters it's acceptable
        for (let i = 0; i < this.clusters.length; i++) {
            const clusterA = this.clusters[i];
            
            for (let j = i + 1; j < this.clusters.length; j++) {
                const clusterB = this.clusters[j];
                
                // Check each pair of nodes between the two clusters
                for (let nodeA of clusterA.nodes) {
                    for (let nodeB of clusterB.nodes) {
                        // Skip if either node is being dragged (has fixed position)
                        if (nodeA.fx !== null || nodeA.fy !== null || 
                            nodeB.fx !== null || nodeB.fy !== null) continue;
                        
                        // Distance between the nodes
                        const dx = nodeB.x - nodeA.x;
                        const dy = nodeB.y - nodeA.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        
                        // Minimum distance required (sum of radii + extra padding)
                        const radiusA = this.getNodeRadius(nodeA);
                        const radiusB = this.getNodeRadius(nodeB);
                        const minDistance = radiusA + radiusB + INTER_CLUSTER_PADDING;
                        
                        // If nodes are too close
                        if (distance < minDistance && distance > 0) {
                            // Calculate repulsion force
                            const force = (minDistance - distance) / distance;
                            
                            // Unit vector for force direction
                            const unitX = dx / distance;
                            const unitY = dy / distance;
                            
                            // Apply stronger forces for inter-cluster collisions
                            // Move both nodes apart proportional to force
                            nodeA.x -= unitX * force * 2; // Doubled strength
                            nodeA.y -= unitY * force * 2;
                            nodeB.x += unitX * force * 2;
                            nodeB.y += unitY * force * 2;
                            
                            // Add velocity to help maintain separation
                            // This makes the nodes "bounce" away from each other
                            if (nodeA.vx !== undefined && nodeA.vy !== undefined) {
                                nodeA.vx -= unitX * force * 0.5;
                                nodeA.vy -= unitY * force * 0.5;
                            }
                            
                            if (nodeB.vx !== undefined && nodeB.vy !== undefined) {
                                nodeB.vx += unitX * force * 0.5;
                                nodeB.vy += unitY * force * 0.5;
                            }
                        }
                    }
                }
            }
        }
    }

    //--------------------------------------
    //  CLUSTER SEPARATION LOGIC
    //--------------------------------------
    updateClusterBounds(clusterId) {
        const cluster = this.getCluster(clusterId);
        if (!cluster || !cluster.nodes.length) return;
        
        // Find min/max coordinates of all nodes in this cluster
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        cluster.nodes.forEach(node => {
            if (!node.x || !node.y) return; // Skip nodes without positions
            
            const radius = this.getNodeRadius(node);
            minX = Math.min(minX, node.x - radius);
            minY = Math.min(minY, node.y - radius);
            maxX = Math.max(maxX, node.x + radius);
            maxY = Math.max(maxY, node.y + radius);
        });
        
        // Update bounds if we have valid values
        if (minX !== Infinity && minY !== Infinity && maxX !== -Infinity && maxY !== -Infinity) {
            cluster.bounds = {
                minX, minY, maxX, maxY,
                // Calculate center point and radius
                centerX: (minX + maxX) / 2,
                centerY: (minY + maxY) / 2,
                radius: Math.max(maxX - minX, maxY - minY) / 2 + 50 // Add padding
            };
        }
    }

    applyClusterSeparationForce() {
        // Enhanced cluster-level separation with more aggressive parameters
        const MIN_SEPARATION = 100; // Increased from 50 to 100
        
        // Loop through all cluster pairs
        for (let i = 0; i < this.clusters.length; i++) {
            const clusterA = this.clusters[i];
            
            for (let j = i + 1; j < this.clusters.length; j++) {
                const clusterB = this.clusters[j];
                
                // Calculate distance between cluster centers
                const dx = clusterB.bounds.centerX - clusterA.bounds.centerX;
                const dy = clusterB.bounds.centerY - clusterA.bounds.centerY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Minimum required distance (increased padding)
                const minDistance = clusterA.bounds.radius + clusterB.bounds.radius + MIN_SEPARATION;
                
                // If clusters are too close
                if (distance < minDistance && distance > 0) {
                    // Unit vector of the direction to push
                    const unitX = dx / distance;
                    const unitY = dy / distance;
                    
                    // Force strength (stronger when closer)
                    const strength = (minDistance - distance) / minDistance * 0.2; // Doubled from 0.1
                    
                    // Apply force to all nodes in both clusters
                    clusterA.nodes.forEach(node => {
                        if (node.fx === null) { // Only apply force to non-fixed nodes
                            node.vx -= unitX * strength * 15; // Increased from 10
                            node.vy -= unitY * strength * 15;
                        }
                    });
                    
                    clusterB.nodes.forEach(node => {
                        if (node.fx === null) { // Only apply force to non-fixed nodes
                            node.vx += unitX * strength * 15;
                            node.vy += unitY * strength * 15;
                        }
                    });
                    
                    // Also apply a direct position adjustment for faster response
                    if (!this.isClusterBeingDragged(clusterA) && !this.isClusterBeingDragged(clusterB)) {
                        const moveAmount = (minDistance - distance) / 2 * 1.5; // 50% stronger movement
                        
                        this.moveCluster(clusterA, -unitX * moveAmount, -unitY * moveAmount);
                        this.moveCluster(clusterB, unitX * moveAmount, unitY * moveAmount);
                    }
                }
            }
        }
    }

    isClusterBeingDragged(cluster) {
        // Return true if any node in the cluster has fx, fy not null (indicating it's being dragged)
        return cluster.nodes.some(node => node.fx !== null || node.fy !== null);
    }

    moveCluster(cluster, dx, dy) {
        // Move all nodes in a cluster by dx, dy
        cluster.nodes.forEach(node => {
            node.x += dx;
            node.y += dy;
        });
        
        // Update visual position immediately
        cluster.nodeGroup.selectAll(".paper-node-group")
            .attr("transform", d => `translate(${d.x},${d.y})`);
            
        // Update link positions
        cluster.linkGroup.selectAll(".paper-link")
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
            
        // Update bounds
        this.updateClusterBounds(cluster.id);
    }

//--------------------------------------
//  DRAG LOGIC
//--------------------------------------
dragstarted(event, d, clusterId) {
    // Prevent zoom during drag
    if (event.sourceEvent) event.sourceEvent.stopPropagation();
    
    const cluster = this.getCluster(clusterId);
    if (!event.active) cluster.simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

dragged(event, d, clusterId) {
    d.fx = event.x;
    d.fy = event.y;
}

async dragended(event, d, clusterId) {
    const cluster = this.getCluster(clusterId);
    d.fx = null;
    d.fy = null;
    cluster.simulation.alpha(0.3).restart();

    // If center node => ignore
    if (d.id === 0) return;

    // Distance from cluster center
    const centerNode = cluster.nodes.find(n => n.id === 0);
    if (!centerNode) return;
    const dist = Math.hypot(d.x - centerNode.x, d.y - centerNode.y);

    if (dist < this.centerThreshold) {
        // => become new center
        this.animateNodeToCenter(d, cluster);
    } else if (dist > this.detachThreshold) {
        // => new cluster
        this.detachNode(d, cluster);
    }
}

getCluster(cid) {
    return this.clusters.find(c => c.id === cid);
}

//--------------------------------------
//  MAKE A NODE THE NEW CENTER - WITH SEAMLESS TRANSITION
//--------------------------------------
animateNodeToCenter(nodeDatum, cluster) {
    // 1. Get exact positions for a seamless transition
    const exactNodePos = { x: nodeDatum.x, y: nodeDatum.y };
    
    // 2. Stop simulation during transition for stability
    cluster.simulation.stop();
    
    // 3. Save the node's visual elements to preserve them during transition
    const chosenNodeSel = cluster.nodeGroup
        .selectAll(".paper-node-group")
        .filter(d => d === nodeDatum);
    
    // 4. Create a transition holder div to maintain the node position during data fetch
    const transitionNodeGroup = this.zoomGroup.append("g")
        .attr("class", "transition-node-group")
        .attr("transform", `translate(${exactNodePos.x},${exactNodePos.y})`);
        
    // 5. Clone the visual appearance of the chosen node
    const transitionCircle = transitionNodeGroup.append("circle")
        .attr("class", "paper-node")
        .attr("r", this.getNodeRadius(nodeDatum))
        .attr("fill", this.getNodeColor(nodeDatum))
        .attr("stroke", "#333")
        .attr("stroke-width", 1.5);
        
    const transitionText = transitionNodeGroup.append("text")
        .attr("class", "paper-title")
        .attr("text-anchor", "middle")
        .attr("dy", ".35em")
        .style("font-size", "11px")
        .style("fill", "#f0f0f0")
        .text(this.truncateTitle(nodeDatum.title, 25));
    
    // 6. Fade out all existing nodes and links EXCEPT the chosen node
    cluster.nodeGroup.selectAll(".paper-node-group")
        .filter(d => d !== nodeDatum)
        .transition()
        .duration(400)
        .style("opacity", 0)
        .remove();
        
    cluster.linkGroup.selectAll(".paper-link")
        .transition()
        .duration(400)
        .style("opacity", 0)
        .remove();
    
    // 7. Expand the transition node
    transitionCircle.transition()
        .duration(600)
        .attr("r", 70)
        .attr("fill", "#ff3333");
        
    transitionText.transition()
        .duration(600)
        .style("font-size", "14px")
        .style("font-weight", "bold")
        .style("fill", "#fff");
    
    // 8. Fade out the original node once we have the transition copy
    chosenNodeSel.transition()
        .duration(300)
        .style("opacity", 0)
        .remove();
    
    // 9. Completely remove the old cluster (but keep our transition node)
    setTimeout(() => {
        if (cluster.group) {
            cluster.group.remove();
        }
        this.clusters = this.clusters.filter(c => c !== cluster);
        
        // 10. Fetch data and create new cluster exactly where the transition node is
        const fetchAndCreateNewCluster = async () => {
            const origId = nodeDatum.original_id;
            if (origId !== undefined) {
                const newData = await this.fetchPaperAsQuery(origId);
                if (newData && newData.center) {
                    // 11. Create the new cluster precisely at the transition node position
                    this.createInPlaceCluster(newData, exactNodePos, transitionNodeGroup);
                }
            }
        };
        
        fetchAndCreateNewCluster();
    }, 500); // Short delay to ensure smooth transition
}

//--------------------------------------
//  CREATE IN-PLACE CLUSTER - ENSURES PERFECT POSITIONING
//--------------------------------------
createInPlaceCluster(mapData, position, transitionNode) {
    // Create a new cluster to replace the transition node
    const cid = ++this.clusterCounter;

    // Use the center data from our fetch but keep the position exact
    const centerNode = { ...mapData.center, __clusterId: cid, id: 0, x: position.x, y: position.y, fx: position.x, fy: position.y };
    const relatedNodes = mapData.related.map((r, i) => ({
        ...r, __clusterId: cid, id: i + 1,
        // Start all related nodes at the center position for animation outward
        x: position.x, y: position.y
    }));
    
    const nodes = [centerNode, ...relatedNodes];
    const links = relatedNodes.map(n => ({
        source: 0,
        target: n.id,
        value: n.score
    }));

    // Create the cluster group
    const clusterGroup = this.zoomGroup.append("g")
        .attr("class", `cluster-group c${cid}`);
    
    const linkGroup = clusterGroup.append("g").attr("class", "links-container");
    const nodeGroup = clusterGroup.append("g").attr("class", "nodes-container");
    
    // Create links with starting point at center
    const linkSel = linkGroup.selectAll("line")
        .data(links)
        .enter()
        .append("line")
        .attr("class", "paper-link")
        .attr("stroke", "#444")
        .attr("stroke-opacity", 0) // Start invisible
        .attr("stroke-width", d => Math.max(1, d.value * 6))
        .attr("x1", position.x)
        .attr("y1", position.y)
        .attr("x2", position.x)
        .attr("y2", position.y);
    
    // Create nodes at center position initially
    const nodeSel = nodeGroup.selectAll("g.paper-node-group")
        .data(nodes)
        .enter()
        .append("g")
        .attr("class", "paper-node-group")
        .attr("transform", d => `translate(${position.x},${position.y})`)
        .style("opacity", d => d.id === 0 ? 1 : 0) // Only center visible initially
        .call(d3.drag()
            .on("start", (event, d) => this.dragstarted(event, d, cid))
            .on("drag", (event, d) => this.dragged(event, d, cid))
            .on("end", (event, d) => this.dragended(event, d, cid))
        );
    
    // Create the circles for each node
    nodeSel.append("circle")
        .attr("class", "paper-node")
        .attr("r", d => this.getNodeRadius(d))
        .attr("fill", d => this.getNodeColor(d))
        .attr("stroke", "#333")
        .attr("stroke-width", 1.5)
        .on("mouseover", (event, d) => this.handleNodeMouseOver(event, d))
        .on("mouseout", () => this.handleNodeMouseOut())
        .on("click", (event, d) => this.openPaperLink(d));
    
    // Add text titles
    this.addNodeTitles(nodeSel);
    
    // Hide the center node initially (we'll reveal it when transition node is removed)
    nodeSel.filter(d => d.id === 0).style("opacity", 0);
    
    // Create the force simulation (gentle start)
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(d => 200 * (1 - d.value)))
        .force("charge", d3.forceManyBody().strength(-1200))
        .force("collision", d3.forceCollide().radius(d => this.getNodeRadius(d) + 25))
        .force("x", d3.forceX(position.x).strength(0.1))
        .force("y", d3.forceY(position.y).strength(0.1))
        .alpha(0); // Start paused
    
    // Keep center node fixed initially
    centerNode.fx = position.x;
    centerNode.fy = position.y;
    
    simulation.on("tick", () => {
        // Apply inter-cluster node collision force
        if (this.clusters.length > 0) {
            this.applyInterClusterNodeCollision();
        }
        
        linkSel
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
        
        nodeSel
            .attr("transform", d => `translate(${d.x},${d.y})`);
        
        this.updateClusterBounds(cid);
        
        if (this.clusters.length > 1) {
            this.applyClusterSeparationForce();
        }
    });
    
    // Save to clusters array
    this.clusters.push({
        id: cid,
        group: clusterGroup,
        linkGroup,
        nodeGroup,
        nodes,
        links,
        simulation,
        position,
        bounds: {
            minX: position.x - 200,
            maxX: position.x + 200,
            minY: position.y - 200,
            maxY: position.y + 200,
            centerX: position.x,
            centerY: position.y,
            radius: 200
        }
    });
    
    this.updateClusterBounds(cid);
    
    // Start the seamless transition
    // 1. First remove transition node while revealing center node
    setTimeout(() => {
        // Fade out transition node
        transitionNode.transition()
            .duration(300)
            .style("opacity", 0)
            .remove();
            
        // Fade in our real center node
        nodeSel.filter(d => d.id === 0)
            .transition()
            .duration(300)
            .style("opacity", 1);
            
        // 2. After center node is visible, begin radiating links and nodes outward
        setTimeout(() => {
            // Release the center node's fixed position gradually
            setTimeout(() => {
                centerNode.fx = null;
                centerNode.fy = null;
            }, 1000);
            
            // Fade in links with staggered timing
            linkSel.transition()
                .delay((d, i) => i * 25)
                .duration(500)
                .attr("stroke-opacity", 0.6);
            
            // Fade in nodes with staggered timing
            nodeSel.filter(d => d.id !== 0)
                .transition()
                .delay((d, i) => i * 30)
                .duration(600)
                .style("opacity", 1);
                
            // Gently start simulation
            simulation.alpha(0.3).restart();
            
            // Gradually increase simulation intensity
            setTimeout(() => {
                simulation.alpha(0.8).restart();
            }, 500);
        }, 300);
    }, 300);
    
    return this.clusters[this.clusters.length - 1];
}

//--------------------------------------
//  DETACH => NEW CLUSTER WITH SEAMLESS ANIMATION
//--------------------------------------
detachNode(nodeDatum, cluster) {
    // 1. Capture exact position for seamless transition
    const exactNodePos = { x: nodeDatum.x, y: nodeDatum.y };
    
    // 2. Create a transition node to maintain visual continuity during data fetch
    const transitionNodeGroup = this.zoomGroup.append("g")
        .attr("class", "transition-node-group")
        .attr("transform", `translate(${exactNodePos.x},${exactNodePos.y})`);
        
    // Clone the visual appearance of the detached node
    const transitionCircle = transitionNodeGroup.append("circle")
        .attr("class", "paper-node")
        .attr("r", this.getNodeRadius(nodeDatum))
        .attr("fill", this.getNodeColor(nodeDatum))
        .attr("stroke", "#333")
        .attr("stroke-width", 1.5);
        
    const transitionText = transitionNodeGroup.append("text")
        .attr("class", "paper-title")
        .attr("text-anchor", "middle")
        .attr("dy", ".35em")
        .style("font-size", "11px")
        .style("fill", "#f0f0f0")
        .text(this.truncateTitle(nodeDatum.title, 25));
        
    // 3. Highlight the transition node briefly to show it's being detached
    transitionCircle.transition()
        .duration(300)
        .attr("r", this.getNodeRadius(nodeDatum) * 1.2)
        .attr("stroke", "#ff3333")
        .attr("stroke-width", 3);
    
    // 4. Find and animate out the connections from the original cluster
    const connectedLinks = cluster.linkGroup
        .selectAll(".paper-link")
        .filter(l => l.source === nodeDatum || l.target === nodeDatum);
    
    // Fade out connected links with transition
    connectedLinks
        .transition()
        .duration(400)
        .style("opacity", 0)
        .remove();

    // 5. Fade out the original node from the cluster
    const nodeSel = cluster.nodeGroup
        .selectAll(".paper-node-group")
        .filter(d => d === nodeDatum);

    nodeSel.transition()
        .duration(400)
        .style("opacity", 0)
        .remove();
        
    // 6. Update the data structures in the original cluster
    cluster.links = cluster.links.filter(l => 
        l.source !== nodeDatum && l.target !== nodeDatum
    );
    cluster.nodes = cluster.nodes.filter(n => n !== nodeDatum);
    
    // 7. Restart the original cluster's simulation
    cluster.simulation.alpha(0.3).restart();

    // 8. If no original_id, we can't fetch new data
    if (nodeDatum.original_id === undefined) {
        // Just clean up the transition node in this case
        setTimeout(() => {
            transitionNodeGroup.transition()
                .duration(400)
                .style("opacity", 0)
                .remove();
        }, 800);
        return;
    }

    // 9. Fetch data for the new cluster while maintaining the transition node
    setTimeout(async () => {
        try {
            const data = await this.fetchPaperAsQuery(nodeDatum.original_id);
            if (!data || !data.center) {
                // If we couldn't get data, fade out the transition node
                transitionNodeGroup.transition()
                    .duration(400)
                    .style("opacity", 0)
                    .remove();
                return;
            }
            
            // 10. Create the new cluster in place with a smooth animation
            this.createInPlaceCluster(data, exactNodePos, transitionNodeGroup);
            
        } catch (err) {
            console.error("Error fetching data for detached node:", err);
            // Clean up the transition node on error
            transitionNodeGroup.transition()
                .duration(400)
                .style("opacity", 0)
                .remove();
        }
    }, 500); // Short delay for visual effect and to ensure links are removed
}

    //--------------------------------------
    //  FETCH PAPER AS QUERY (no entire re-render)
    //--------------------------------------
    async fetchPaperAsQuery(originalId) {
        try {
            const resp = await fetch("/paper_as_query?" + new URLSearchParams({ paper_id: originalId }).toString());
            if (!resp.ok) throw new Error(`Network response not ok: ${resp.statusText}`);
            const data = await resp.json();
            if (!data.center) return null;
            return data;
        } catch (err) {
            console.error("fetchPaperAsQuery error:", err);
            return null;
        }
    }

    //--------------------------------------
    //  NODES: RADIUS, COLOR, TITLE, etc.
    //--------------------------------------
    getNodeRadius(d) {
        // cluster center is bigger
        if (d.id === 0) return 70;
        return 30 + (d.score * 25);
    }

    getNodeColor(d) {
        if (d.id === 0) return "#ff3333";
        if (d.score > 0.75) return "#ff6666";
        if (d.score > 0.5) return "#992222";
        return "#661111";
    }

    addNodeTitles(selection) {
        selection.append("text")
            .attr("class", "paper-title")
            .attr("text-anchor", "middle")
            .attr("dy", ".35em")
            .style("font-size", d => d.id === 0 ? "14px" : "11px")
            .style("fill", d => d.id === 0 ? "#fff" : "#f0f0f0")
            .style("font-weight", d => d.id === 0 ? "bold" : "normal")
            .text(d => this.truncateTitle(d.title, 25));
    }

    truncateTitle(title, maxLen) {
        if (!title) return "";
        return title.length > maxLen ? title.slice(0, maxLen) + "..." : title;
    }

    //--------------------------------------
    //  TOOLTIP (DELAY)
    //--------------------------------------
    handleNodeMouseOver(event, d) {
        clearTimeout(this.hoverTimer);
        this.hoverTimer = setTimeout(() => {
            const pageX = event.pageX, pageY = event.pageY;
            const tipW = 300;
            let leftPos = pageX + 15;
            if (leftPos + tipW > window.innerWidth) {
                leftPos = pageX - tipW - 15;
            }
            this.tooltip.style.left = leftPos + "px";
            this.tooltip.style.top = (pageY - 100) + "px";

            // Fill contents
            this.tooltip.querySelector(".tooltip-title").textContent = d.title;
            this.tooltip.querySelector(".tooltip-abstract").textContent = d.abstract || "No abstract available";

            // Show
            this.tooltip.style.display = "block";
            
            // Store the current node for reference
            this.tooltip.dataset.currentNode = d.id;
            
            // Add mouseenter event to the tooltip
            this.tooltip.addEventListener('mouseenter', this.handleTooltipMouseEnter.bind(this));
            this.tooltip.addEventListener('mouseleave', this.handleTooltipMouseLeave.bind(this));
        }, this.tooltipDelay);
    }

    handleNodeMouseOut() {
        clearTimeout(this.hoverTimer);
        
        // Don't hide immediately, give time to potentially enter the tooltip
        this.hoverTimer = setTimeout(() => {
            // Only hide if mouse is not over tooltip
            if (!this.isMouseOverTooltip) {
                this.tooltip.style.display = "none";
            }
        }, 100);
    }

    // Add these new methods
    handleTooltipMouseEnter() {
        this.isMouseOverTooltip = true;
        clearTimeout(this.hoverTimer);
    }

    handleTooltipMouseLeave() {
        this.isMouseOverTooltip = false;
        this.tooltip.style.display = "none";
        
        // Remove the event listeners to prevent memory leaks
        this.tooltip.removeEventListener('mouseenter', this.handleTooltipMouseEnter);
        this.tooltip.removeEventListener('mouseleave', this.handleTooltipMouseLeave);
    }

    //------------------ 
    // POPUP 
    //-------------------
    openPaperLink(d) {
        // Instead of opening a new window, open the PDF in a popup:
        this.createPaperPopup(d.link, d.title);
    }

    createPaperPopup(pdfUrl, paperTitle) {
        // 1. Create a semi-transparent overlay
        const overlay = document.createElement('div');
        overlay.className = 'paper-popup-container';

        // 2. Create the popup content wrapper
        const popupContent = document.createElement('div');
        popupContent.className = 'paper-popup-content';

        // 3. Header with close button
        const header = document.createElement('div');
        header.className = 'paper-popup-header';

        const titleEl = document.createElement('h3');
        titleEl.className = 'paper-popup-title';
        titleEl.textContent = paperTitle || "Paper PDF";

        const closeBtn = document.createElement('button');
        closeBtn.className = 'paper-popup-close';
        closeBtn.innerHTML = '&times;';
        closeBtn.onclick = () => {
            document.body.removeChild(overlay);
        };

        header.appendChild(titleEl);
        header.appendChild(closeBtn);

        // 4. Body with the PDF in an iframe
        const body = document.createElement('div');
        body.className = 'paper-popup-body';

        const iframe = document.createElement('iframe');
        iframe.className = 'paper-popup-iframe';
        iframe.src = pdfUrl; // Link to your PDF (arXiv link, etc.)

        body.appendChild(iframe);

        // 5. Footer (optional: external link or other controls)
        const footer = document.createElement('div');
        footer.className = 'paper-popup-footer';

        const externalLink = document.createElement('a');
        externalLink.className = 'paper-popup-external-link';
        externalLink.href = pdfUrl;
        externalLink.target = '_blank';
        externalLink.textContent = 'Open in new tab';
        
        footer.appendChild(externalLink);

        // 6. Put it all together
        popupContent.appendChild(header);
        popupContent.appendChild(body);
        popupContent.appendChild(footer);
        overlay.appendChild(popupContent);

        // 7. Add overlay to document
        document.body.appendChild(overlay);
    }

}

//--------------------------------------
//  INIT ON DOMContentLoaded
//--------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    // Instantiate
    const multiMap = new MultiClusterMap('map-container', 'paper-tooltip');

    // Basic input handling
    const searchInput = document.getElementById('map-search-input');

    function debounce(fn, wait) {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), wait);
        };
    }

    const debouncedSearch = debounce(() => {
        const query = searchInput.value.trim();
        if (query) multiMap.fetchData(query);
    }, 500);

    searchInput.addEventListener('input', debouncedSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = searchInput.value.trim();
            if (query) multiMap.fetchData(query);
        }
    });

    // If you have an onclick to focus
    window.sendFocus = function() {
        searchInput.focus();
    };

    // Optionally auto-focus
    searchInput.focus();
});
