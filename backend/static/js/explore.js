// explore.js - Advanced visualization for Research Papers using D3.js

document.addEventListener('DOMContentLoaded', function() {
    // Constants and Configuration
    const GALAXY_CONFIG = {
        width: 0,  // Will be set based on container size
        height: 0, // Will be set based on container size
        nodeRadius: 5,
        centerNodeRadius: 12,
        linkDistance: 120,
        charge: -400,
        centerForce: 0.03,
        clusters: {
            'Machine Learning': { color: '#FF5555', x: 0.3, y: 0.3 },
            'Computer Vision': { color: '#55AAFF', x: 0.7, y: 0.3 },
            'NLP': { color: '#FFAA55', x: 0.3, y: 0.7 },
            'Reinforcement Learning': { color: '#55FF55', x: 0.7, y: 0.7 },
            'Generative Models': { color: '#AA55FF', x: 0.5, y: 0.5 }
        },
        maxPapers: 500, // Maximum papers to show
        journeyHighlightColor: '#FFFFFF',
        backgroundColor: '#121212',
        starColors: ['#FFFFFF', '#AAAAAA', '#888888', '#666666', '#444444'],
        starCount: 500
    };

    // State Management
    const state = {
        papers: [],
        journeys: [],
        simulationRunning: false,
        selectedPaper: null,
        activeJourney: null,
        currentJourneyStep: 0,
        viewMode: 'galaxy', // 'galaxy' or 'cluster'
        nodeSize: 10,
        zoom: null,
        simulation: null,
        filters: {
            domains: []
        },
        tooltipVisible: false,
        loading: true,
        svg: null,
        container: null,
        nodes: [],
        links: [],
        starNodes: [],
        initialDataLoaded: false,
        activeJourneyHighlight: null, // Track if a journey is actively highlighted
    };

    // DOM Elements
    const elements = {
        galaxyContainer: document.getElementById('galaxy-container'),
        loadingContainer: document.getElementById('loading-container'),
        paperDetail: document.getElementById('paper-detail'),
        closePanel: document.getElementById('close-panel'),
        paperTitle: document.getElementById('paper-title'),
        paperAbstract: document.getElementById('paper-abstract'),
        citationCount: document.getElementById('citation-count'),
        publicationYear: document.getElementById('publication-year'),
        readingLevel: document.getElementById('reading-level'),
        paperLink: document.getElementById('paper-link'),
        relatedPapersList: document.getElementById('related-papers-list'),
        domainFilters: document.getElementById('domain-filters'),
        journeyList: document.getElementById('journey-list'),
        journeyGuide: document.getElementById('journey-guide'),
        closeJourney: document.getElementById('close-journey'),
        journeyTitle: document.getElementById('journey-title'),
        journeyDescription: document.getElementById('journey-description'),
        journeyProgressFill: document.getElementById('journey-progress-fill'),
        currentStep: document.getElementById('current-step'),
        totalSteps: document.getElementById('total-steps'),
        prevPaper: document.getElementById('prev-paper'),
        nextPaper: document.getElementById('next-paper'),
        journeyPaperTitle: document.getElementById('journey-paper-title'),
        journeyPaperAbstract: document.getElementById('journey-paper-abstract'),
        journeyPaperLink: document.getElementById('journey-paper-link'),
        zoomIn: document.getElementById('zoom-in'),
        zoomOut: document.getElementById('zoom-out'),
        resetView: document.getElementById('reset-view'),
        galaxyView: document.getElementById('galaxy-view'),
        clusterView: document.getElementById('cluster-view'),
        nodeSize: document.getElementById('node-size'),
        startExploring: document.getElementById('start-exploring'),
        tooltipContent: document.getElementById('tooltip-content'),
    };

    // ----- DATA LOADING AND INITIALIZATION -----

    // Fetch initial data to bootstrap the visualization

    async function fetchInitialData() {
        try {
            // In a real implementation, this would be a call to an API endpoint
            const response = await fetch('/explore_data');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            state.papers = data.papers || [];
            state.journeys = data.journeys || [];
            
            console.log(`Loaded ${state.papers.length} papers and ${state.journeys.length} journeys`);
            state.initialDataLoaded = true;
            
            // Calculate initial positions for papers based on their clusters
            state.papers.forEach(paper => {
                const clusterData = GALAXY_CONFIG.clusters[paper.cluster] || GALAXY_CONFIG.clusters['Machine Learning'];
                paper.x = (clusterData.x + (Math.random() * 0.2 - 0.1)) * GALAXY_CONFIG.width;
                paper.y = (clusterData.y + (Math.random() * 0.2 - 0.1)) * GALAXY_CONFIG.height;
            });
            
            // Once data is loaded, initialize the visualization
            initializeVisualization();
            populateDomainFilters();
            populateJourneyList();
            hideLoading();
        } catch (error) {
            console.error('Error fetching initial data:', error);
            // Show error message
            elements.loadingContainer.innerHTML = `
                <div class="error">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>Failed to load research data: ${error.message}</p>
                    <button id="retry-loading" class="paper-link">Retry</button>
                </div>
            `;
            document.getElementById('retry-loading').addEventListener('click', () => {
                location.reload();
            });
        }
    }

    // ----- VISUALIZATION INITIALIZATION -----
    // Main visualization initialization
    function initializeVisualization() {
        // Set dimensions based on container size
        const containerRect = elements.galaxyContainer.getBoundingClientRect();
        GALAXY_CONFIG.width = containerRect.width;
        GALAXY_CONFIG.height = containerRect.height;

        // Create SVG container
        state.svg = d3.select('#galaxy-container')
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', [0, 0, GALAXY_CONFIG.width, GALAXY_CONFIG.height]);

        // Create background
        state.svg.append('rect')
            .attr('width', GALAXY_CONFIG.width)
            .attr('height', GALAXY_CONFIG.height)
            .attr('fill', '#0a0a0a')
            .attr('id', 'visualization-background');

        // Create container for all elements (for zoom)
        state.container = state.svg.append('g')
            .attr('class', 'galaxy-container');

        // Generate stars for the background (uses extendedFactor 6)
        generateStars();

        // Initialize D3 force simulation
        initializeForceSimulation();

        // Add event listeners for controls
        addEventListeners();

        // Define the extended area parameters using the same extendedFactor as generateStars
        const extendedFactor = 6; // Must match generateStars extendedFactor
        const extendedWidth = GALAXY_CONFIG.width * extendedFactor;
        const extendedHeight = GALAXY_CONFIG.height * extendedFactor;
        const offsetX = (extendedWidth - GALAXY_CONFIG.width) / 2;
        const offsetY = (extendedHeight - GALAXY_CONFIG.height) / 2;

        // Create zoom behavior with proper panning limits
        state.zoom = d3.zoom()
            .scaleExtent([0.2, 5])
            .translateExtent([[-offsetX, -offsetY], [GALAXY_CONFIG.width + offsetX, GALAXY_CONFIG.height + offsetY]])
            .on('zoom', zoomed);

        // Attach the zoom behavior to the SVG (only one initialization)
        state.svg.call(state.zoom);
    }

    function highlightJourneyPath(journey, isPermanent = false) {
        // Store active highlight if permanent
        if (isPermanent) {
            state.activeJourneyHighlight = journey.id;
        }
        
        // Get all nodes in this journey
        const journeyNodeIds = journey.steps;
        const journeyNodes = journeyNodeIds.map(id => state.nodes.find(n => n.id === id));
        
        // Fade all nodes
        d3.selectAll('.paper-node')
            .transition()
            .duration(300)
            .style('opacity', 0.2);
        
        // Highlight journey nodes
        journeyNodeIds.forEach(nodeId => {
            d3.select(`#node-${nodeId}`)
                .transition()
                .duration(300)
                .style('opacity', 1);
                
            // Make journey nodes glow
            d3.select(`#node-${nodeId}`).select('circle')
                .transition()
                .duration(300)
                .attr('r', d => d.radius * 1.5)
                .attr('stroke', journey.class === 'ml' ? '#FF5555' : 
                                 journey.class === 'cv' ? '#55AAFF' :
                                 journey.class === 'nlp' ? '#FFAA55' :
                                 journey.class === 'rl' ? '#55FF55' : '#AA55FF')
                .attr('stroke-width', 3);
                
            d3.select(`#node-${nodeId}`).select('.node-glow')
                .transition()
                .duration(300)
                .attr('r', d => d.radius * 2.5)
                .attr('opacity', 0.4);
        });
        
        // Remove any existing journey path
        state.container.selectAll('.journey-path').remove();
        
        // Create path data between consecutive nodes
        const pathData = [];
        for (let i = 0; i < journeyNodes.length - 1; i++) {
            const source = journeyNodes[i];
            const target = journeyNodes[i + 1];
            
            if (source && target) {
                pathData.push({
                    source,
                    target
                });
            }
        }
        
        // Draw the journey path with brighter edges
        state.container.append('g')
            .attr('class', 'journey-path')
            .selectAll('line')
            .data(pathData)
            .enter()
            .append('line')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y)
            .attr('stroke', journey.class === 'ml' ? '#FF5555' : 
                             journey.class === 'cv' ? '#55AAFF' :
                             journey.class === 'nlp' ? '#FFAA55' :
                             journey.class === 'rl' ? '#55FF55' : '#AA55FF')
            .attr('stroke-opacity', 0.8)
            .attr('stroke-width', 3);
    }
    
    // Function to reset highlighting
    function resetJourneyHighlight() {
        // Clear active highlight
        state.activeJourneyHighlight = null;
        
        // Reset all nodes opacity
        d3.selectAll('.paper-node')
            .transition()
            .duration(300)
            .style('opacity', 1);
            
        // Reset all nodes appearance
        d3.selectAll('.paper-node circle')
            .transition()
            .duration(300)
            .attr('r', d => d.radius)
            .attr('stroke', '#222')
            .attr('stroke-width', 1);
            
        d3.selectAll('.paper-node .node-glow')
            .transition()
            .duration(300)
            .attr('r', d => d.radius * 1.5)
            .attr('opacity', 0.2);
        
        // Remove journey path
        state.container.selectAll('.journey-path')
            .transition()
            .duration(300)
            .style('opacity', 0)
            .remove();
    }

    // Initialize the force simulation with nodes and links
    function initializeForceSimulation() {
        // Create nodes from paper data
        state.nodes = state.papers.map(paper => ({
            id: paper.id,
            radius: GALAXY_CONFIG.nodeRadius,
            cluster: paper.cluster,
            color: GALAXY_CONFIG.clusters[paper.cluster].color,
            x: paper.x || (GALAXY_CONFIG.width / 2 + (Math.random() - 0.5) * 200),
            y: paper.y || (GALAXY_CONFIG.height / 2 + (Math.random() - 0.5) * 200),
            paper: paper
        }));
        
        // Create links based on related papers
        state.links = [];
        state.papers.forEach(paper => {
            paper.related.forEach(relatedId => {
                state.links.push({
                    source: paper.id,
                    target: relatedId,
                    value: 1
                });
            });
        });
        
        // Create force simulation
        state.simulation = d3.forceSimulation(state.nodes)
            .force('charge', d3.forceManyBody().strength(GALAXY_CONFIG.charge))
            .force('link', d3.forceLink(state.links).id(d => d.id).distance(GALAXY_CONFIG.linkDistance))
            .force('x', d3.forceX().strength(GALAXY_CONFIG.centerForce).x(d => {
                const clusterData = GALAXY_CONFIG.clusters[d.cluster];
                return clusterData.x * GALAXY_CONFIG.width;
            }))
            .force('y', d3.forceY().strength(GALAXY_CONFIG.centerForce).y(d => {
                const clusterData = GALAXY_CONFIG.clusters[d.cluster];
                return clusterData.y * GALAXY_CONFIG.height;
            }))
            .force('collision', d3.forceCollide().radius(d => d.radius * 1.5))
            .on('tick', ticked);
        
        // Create the visual elements
        createLinks();
        createNodes();
    }

    // Create star background
    function generateStars() {
        state.starNodes = [];
        
        // Define an extended area factor (adjust as needed)
        const extendedFactor = 6;
        const extendedWidth = GALAXY_CONFIG.width * extendedFactor;
        const extendedHeight = GALAXY_CONFIG.height * extendedFactor;
        const offsetX = (extendedWidth - GALAXY_CONFIG.width) / 2;
        const offsetY = (extendedHeight - GALAXY_CONFIG.height) / 2;
        
        // Generate random stars in the extended area
        for (let i = 0; i < GALAXY_CONFIG.starCount; i++) {
            state.starNodes.push({
                x: Math.random() * extendedWidth - offsetX,
                y: Math.random() * extendedHeight - offsetY,
                radius: Math.random() * 1.5,
                color: GALAXY_CONFIG.starColors[Math.floor(Math.random() * GALAXY_CONFIG.starColors.length)],
                blink: Math.random() > 0.7
            });
        }
        
        // Create star elements
        const stars = state.container.append('g')
            .attr('class', 'star-container')
            .selectAll('circle')
            .data(state.starNodes)
            .enter()
            .append('circle')
            .attr('class', d => d.blink ? 'star blinking' : 'star')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', d => d.radius)
            .attr('fill', d => d.color)
            .style('opacity', d => d.blink ? 0.7 : 1);
        
        // Add blinking animation to some stars
        stars.filter(d => d.blink)
            .each(function(d) {
                const duration = 1000 + Math.random() * 3000;
                const delay = Math.random() * 3000;
                
                d3.select(this)
                    .transition()
                    .delay(delay)
                    .duration(duration)
                    .style('opacity', 0.3)
                    .transition()
                    .duration(duration)
                    .style('opacity', 0.9)
                    .on('end', function repeat() {
                        d3.select(this)
                            .transition()
                            .duration(duration)
                            .style('opacity', 0.3)
                            .transition()
                            .duration(duration)
                            .style('opacity', 0.9)
                            .on('end', repeat);
                    });
            });
    }

    // Create link elements between paper nodes
    function createLinks() {
        state.container.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(state.links)
            .enter()
            .append('line')
            .attr('stroke', '#444')
            .attr('stroke-opacity', 0.5)
            .attr('stroke-width', 1);
    }

    // Create node elements for papers
    function createNodes() {
        // Create groups for each node
        const nodeGroups = state.container.append('g')
            .attr('class', 'nodes')
            .selectAll('.paper-node')
            .data(state.nodes)
            .enter()
            .append('g')
            .attr('class', 'paper-node')
            .attr('id', d => `node-${d.id}`)
            .on('mouseover', handleNodeMouseOver)
            .on('mouseout', handleNodeMouseOut)
            .on('click', handleNodeClick);
        
        // Add circles for each node
        nodeGroups.append('circle')
            .attr('r', d => d.radius)
            .attr('fill', d => d.color)
            .attr('stroke', '#222')
            .attr('stroke-width', 1);
        
        // Add subtle glow effect
        nodeGroups.append('circle')
            .attr('r', d => d.radius * 1.5)
            .attr('fill', d => d.color)
            .attr('opacity', 0.2)
            .attr('class', 'node-glow');
        
        // Create tooltips for hover info
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'node-tooltip')
            .style('opacity', 0);
        
        // Apply initial animations
        nodeGroups
            .style('opacity', 0)
            .transition()
            .delay((d, i) => i * 10)
            .duration(800)
            .style('opacity', 1);
    }

    // ----- INTERACTION HANDLERS -----

    // Handle mouse hover on paper nodes
    function handleNodeMouseOver(event, d) {
        if (state.tourActive) return;
        
        d3.select(this).select('circle')
            .transition()
            .duration(200)
            .attr('r', d.radius * 1.3);
        
        d3.select(this).select('.node-glow')
            .transition()
            .duration(200)
            .attr('r', d.radius * 2)
            .attr('opacity', 0.3);
        
        // Show tooltip
        const tooltip = d3.select('.node-tooltip');
        if (!tooltip.empty()) {
            tooltip
                .html(`
                    <div class="tooltip-title">${d.paper.title}</div>
                    <div class="tooltip-year">${d.paper.year}</div>
                `)
                .style('left', (event.pageX) + 'px')
                .style('top', (event.pageY - 10) + 'px')
                .transition()
                .duration(200)
                .style('opacity', 1);
        }
        
        // Highlight connected nodes
        d3.selectAll('.paper-node')
            .style('opacity', 0.4);
        
        d3.select(this)
            .style('opacity', 1);
        
        // Find connected nodes and highlight them
        const connectedLinks = state.links.filter(link => 
            link.source.id === d.id || link.target.id === d.id
        );
        
        connectedLinks.forEach(link => {
            const connectedId = link.source.id === d.id ? link.target.id : link.source.id;
            d3.select(`#node-${connectedId}`)
                .style('opacity', 1);
        });
    }

    // Handle mouse out from paper nodes
    function handleNodeMouseOut(event, d) {
        if (state.tourActive) return;
        
        d3.select(this).select('circle')
            .transition()
            .duration(200)
            .attr('r', d.radius);
        
        d3.select(this).select('.node-glow')
            .transition()
            .duration(200)
            .attr('r', d.radius * 1.5)
            .attr('opacity', 0.2);
        
        // Hide tooltip
        const tooltip = d3.select('.node-tooltip');
        if (!tooltip.empty()) {
            tooltip.transition()
                .duration(200)
                .style('opacity', 0);
        }
        
        // Reset all nodes opacity
        d3.selectAll('.paper-node')
            .style('opacity', 1);
    }

    // Handle click on paper nodes
    function handleNodeClick(event, d) {
        if (state.tourActive) return;
        
        state.selectedPaper = d.paper;
        showPaperDetails(d.paper);
        
        // Visually highlight selected node
        d3.selectAll('.paper-node').classed('highlighted', false);
        d3.select(this).classed('highlighted', true);
        
        // If part of active journey, update journey step
        if (state.activeJourney) {
            const stepIndex = state.activeJourney.steps.indexOf(d.paper.id);
            if (stepIndex !== -1) {
                state.currentJourneyStep = stepIndex;
                updateJourneyGuide();
            }
        }
    }

    // Show paper details in the side panel
    function showPaperDetails(paper) {
        elements.paperTitle.textContent = paper.title;
        elements.paperAbstract.textContent = paper.abstract;
        elements.citationCount.textContent = paper.citations;
        elements.publicationYear.textContent = paper.year;
        elements.readingLevel.textContent = paper.readingLevel;
        
        // Change this line to make the link open our popup instead
        elements.paperLink.href = "javascript:void(0)";
        elements.paperLink.setAttribute("data-paper-link", paper.link);
        elements.paperLink.setAttribute("data-paper-title", paper.title);
        
        // Change the icon to indicate popup instead of external link
        const linkIcon = elements.paperLink.querySelector("i");
        if (linkIcon) {
            linkIcon.className = "fas fa-file-pdf"; // Change icon to PDF
        }
        
        // Populate related papers
        elements.relatedPapersList.innerHTML = '';
        const relatedPapers = paper.related.map(id => state.papers.find(p => p.id === id));
        
        relatedPapers.forEach(related => {
            if (!related) return;
            
            const relatedItem = document.createElement('div');
            relatedItem.className = 'related-paper-item';
            relatedItem.innerHTML = `
                <div class="related-paper-title">${related.title}</div>
            `;
            
            relatedItem.addEventListener('click', () => {
                const node = state.nodes.find(n => n.id === related.id);
                if (node) {
                    // Center view on this node
                    centerOnNode(node);
                    // Show its details
                    state.selectedPaper = related;
                    showPaperDetails(related);
                    
                    // Update visual highlight
                    d3.selectAll('.paper-node').classed('highlighted', false);
                    d3.select(`#node-${related.id}`).classed('highlighted', true);
                }
            });
            
            elements.relatedPapersList.appendChild(relatedItem);
        });
        
        // Show the panel
        elements.paperDetail.classList.add('active');
    }

    // Update the journey guide panel for the current step
    function updateJourneyGuide() {
        if (!state.activeJourney) return;
        
        const journey = state.activeJourney;
        const currentPaperId = journey.steps[state.currentJourneyStep];
        const currentPaper = state.papers.find(p => p.id === currentPaperId);
        
        if (!currentPaper) return;
        
        // Update progress indicator
        elements.currentStep.textContent = state.currentJourneyStep + 1;
        const progressPercent = ((state.currentJourneyStep + 1) / journey.steps.length) * 100;
        elements.journeyProgressFill.style.width = `${progressPercent}%`;
        
        // Update paper info
        elements.journeyPaperTitle.textContent = currentPaper.title;
        elements.journeyPaperAbstract.textContent = currentPaper.abstract;
        elements.journeyPaperLink.href = currentPaper.link;
        
        // Update navigation buttons
        elements.prevPaper.disabled = state.currentJourneyStep === 0;
        elements.nextPaper.disabled = state.currentJourneyStep === journey.steps.length - 1;
        
        // Center view on current paper
        const node = state.nodes.find(n => n.id === currentPaperId);
        if (node) {
            centerOnNode(node);
            
            // Also select this node
            state.selectedPaper = currentPaper;
            showPaperDetails(currentPaper);
            
            // Highlight the node
            d3.selectAll('.paper-node').classed('highlighted', false);
            d3.select(`#node-${currentPaperId}`).classed('highlighted', true);
        }
    }

    // Highlight the current journey step
    function highlightJourneyStep() {
        if (!state.activeJourney) return;
        
        // Reset previous highlights
        d3.selectAll('.paper-node').classed('highlighted', false);
        
        // Get current paper ID
        const currentPaperId = state.activeJourney.steps[state.currentJourneyStep];
        
        // Highlight the node
        d3.select(`#node-${currentPaperId}`).classed('highlighted', true);
        
        // Create a path visualization
        updateJourneyPath();
    }

    // Update the visual path for the journey
    function updateJourneyPath() {
        // Remove any existing journey path
        state.container.selectAll('.journey-path').remove();
        
        if (!state.activeJourney) return;
        
        const journey = state.activeJourney;
        const journeyNodes = journey.steps.map(id => state.nodes.find(n => n.id === id));
        
        // Create path data
        const pathData = [];
        for (let i = 0; i < journeyNodes.length - 1; i++) {
            const source = journeyNodes[i];
            const target = journeyNodes[i + 1];
            
            if (source && target) {
                pathData.push({
                    source,
                    target,
                    completed: i < state.currentJourneyStep
                });
            }
        }
        
        // Draw the journey path
        state.container.append('g')
            .attr('class', 'journey-path')
            .selectAll('line')
            .data(pathData)
            .enter()
            .append('line')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y)
            .attr('stroke', d => d.completed ? GALAXY_CONFIG.journeyHighlightColor : 'rgba(255, 255, 255, 0.3)')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', d => d.completed ? 'none' : '5,5');
        
        // Draw step numbers
        state.container.select('.journey-path')
            .selectAll('circle.step-marker')
            .data(journeyNodes)
            .enter()
            .append('circle')
            .attr('class', 'step-marker')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
            .attr('r', 12)
            .attr('fill', (d, i) => i <= state.currentJourneyStep ? 
                  GALAXY_CONFIG.journeyHighlightColor : 'rgba(255, 255, 255, 0.2)')
            .attr('stroke', GALAXY_CONFIG.journeyHighlightColor)
            .attr('stroke-width', 2);
        
        state.container.select('.journey-path')
            .selectAll('text.step-number')
            .data(journeyNodes)
            .enter()
            .append('text')
            .attr('class', 'step-number')
            .attr('x', d => d.x)
            .attr('y', d => d.y + 5)
            .attr('text-anchor', 'middle')
            .attr('fill', (d, i) => i <= state.currentJourneyStep ? '#000' : '#fff')
            .text((d, i) => i + 1);
    }

    // Navigate to next paper in the journey
    function nextJourneyStep() {
        if (!state.activeJourney) return;
        
        if (state.currentJourneyStep < state.activeJourney.steps.length - 1) {
            state.currentJourneyStep++;
            updateJourneyGuide();
            highlightJourneyStep();
        }
    }

    // Navigate to previous paper in the journey
    function prevJourneyStep() {
        if (!state.activeJourney) return;
        
        if (state.currentJourneyStep > 0) {
            state.currentJourneyStep--;
            updateJourneyGuide();
            highlightJourneyStep();
        }
    }

    // Center the view on a specific node
    function centerOnNode(node) {
        const scale = state.zoom.scale();
        const x = -node.x * scale + GALAXY_CONFIG.width / 2;
        const y = -node.y * scale + GALAXY_CONFIG.height / 2;
        
        state.svg.transition()
            .duration(750)
            .call(state.zoom.transform, d3.zoomIdentity.translate(x, y).scale(scale));
    }

    // ----- UI POPULATION FUNCTIONS -----

    // Populate domain filters
    function populateDomainFilters() {
        const domains = Object.keys(GALAXY_CONFIG.clusters);
        
        // Create filter buttons
        elements.domainFilters.innerHTML = '';
        
        domains.forEach(domain => {
            const clusterData = GALAXY_CONFIG.clusters[domain];
            const button = document.createElement('div');
            button.className = 'domain-filter';
            button.textContent = domain;
            button.dataset.domain = domain;
            button.style.borderColor = clusterData.color;
            
            button.addEventListener('click', () => {
                toggleDomainFilter(domain, button);
            });
            
            elements.domainFilters.appendChild(button);
        });
        
        // Add "All" filter
        const allButton = document.createElement('div');
        allButton.className = 'domain-filter active';
        allButton.textContent = 'All Domains';
        allButton.dataset.domain = 'all';
        
        allButton.addEventListener('click', () => {
            // Clear all filters and select "All"
            state.filters.domains = [];
            
            // Update UI
            document.querySelectorAll('.domain-filter').forEach(btn => {
                btn.classList.remove('active');
            });
            allButton.classList.add('active');
            
            // Update visualization
            updateFilters();
        });
        
        elements.domainFilters.prepend(allButton);
    }

    // Toggle domain filter selection
    function toggleDomainFilter(domain, button) {
        const index = state.filters.domains.indexOf(domain);
        const allButton = document.querySelector('.domain-filter[data-domain="all"]');
        
        // Remove "All" selection
        allButton.classList.remove('active');
        
        if (index === -1) {
            // Add this domain to filters
            state.filters.domains.push(domain);
            button.classList.add('active');
        } else {
            // Remove this domain from filters
            state.filters.domains.splice(index, 1);
            button.classList.remove('active');
        }
        
        // If no domains selected, activate "All"
        if (state.filters.domains.length === 0) {
            allButton.classList.add('active');
        }
        
        // Update visualization
        updateFilters();
    }

    // Apply current filters to the visualization
    function updateFilters() {
        // If no domain filters, show all
        if (state.filters.domains.length === 0) {
            d3.selectAll('.paper-node').style('display', 'block');
            return;
        }
        
        // Hide nodes not in selected domains
        d3.selectAll('.paper-node').style('display', d => {
            return state.filters.domains.includes(d.cluster) ? 'block' : 'none';
        });
    }

    // Populate journey list
    function populateJourneyList() {
        elements.journeyList.innerHTML = '';
        
        state.journeys.forEach(journey => {
            const journeyElement = document.createElement('div');
            journeyElement.className = `journey-item ${journey.class}`;
            journeyElement.innerHTML = `
                <div class="journey-title">${journey.title}</div>
                <div class="journey-desc">${journey.description}</div>
            `;
            
            // Add hover effects
            journeyElement.addEventListener('mouseenter', () => {
                // Don't apply hover effect if this journey is already clicked/active
                if (state.activeJourneyHighlight === journey.id) return;
                
                highlightJourneyPath(journey);
            });
            
            journeyElement.addEventListener('mouseleave', () => {
                // Only reset if no journey is actively highlighted
                if (!state.activeJourneyHighlight) {
                    resetJourneyHighlight();
                } else if (state.activeJourneyHighlight !== journey.id) {
                    // If different journey is highlighted, re-highlight it
                    const activeJourney = state.journeys.find(j => j.id === state.activeJourneyHighlight);
                    if (activeJourney) {
                        highlightJourneyPath(activeJourney, true);
                    }
                }
            });
            
            // Add click to toggle permanent highlight
            journeyElement.addEventListener('click', () => {
                // If this journey is already highlighted, reset
                if (state.activeJourneyHighlight === journey.id) {
                    resetJourneyHighlight();
                    
                    // Remove active class from all journey items
                    document.querySelectorAll('.journey-item').forEach(item => {
                        item.classList.remove('active-journey');
                    });
                } else {
                    // Highlight this journey permanently
                    highlightJourneyPath(journey, true);
                    
                    // Add active class to this journey item
                    document.querySelectorAll('.journey-item').forEach(item => {
                        item.classList.remove('active-journey');
                    });
                    journeyElement.classList.add('active-journey');
                }
            });
            
            elements.journeyList.appendChild(journeyElement);
        });
        
        // Add click handler to the background to clear highlighting
        state.svg.on('click', () => {
            if (state.activeJourneyHighlight) {
                resetJourneyHighlight();
                
                // Remove active class from all journey items
                document.querySelectorAll('.journey-item').forEach(item => {
                    item.classList.remove('active-journey');
                });
            }
        });
    }

    // ----- UTILITY FUNCTIONS -----

    // D3 force simulation tick handler
    function ticked() {
        // Update link positions
        state.container.selectAll('.links line')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        // Update node positions
        state.container.selectAll('.paper-node')
            .attr('transform', d => `translate(${d.x}, ${d.y})`);
        
        // Update journey path if active
        if (state.activeJourney) {
            updateJourneyPath();
        }
    }

    // Handle zoom events
    function zoomed(event) {
        state.container.attr('transform', event.transform);
    }

    // Toggle between visualization modes
    function toggleViewMode(mode) {
        if (mode === state.viewMode) return;
        
        state.viewMode = mode;
        elements.galaxyView.classList.toggle('active', mode === 'galaxy');
        elements.clusterView.classList.toggle('active', mode === 'cluster');
        
        // Update forces based on view mode
        if (mode === 'galaxy') {
            // More spread out, cluster-based layout
            state.simulation
                .force('x', d3.forceX().strength(GALAXY_CONFIG.centerForce).x(d => {
                    const clusterData = GALAXY_CONFIG.clusters[d.cluster];
                    return clusterData.x * GALAXY_CONFIG.width;
                }))
                .force('y', d3.forceY().strength(GALAXY_CONFIG.centerForce).y(d => {
                    const clusterData = GALAXY_CONFIG.clusters[d.cluster];
                    return clusterData.y * GALAXY_CONFIG.height;
                }));
        } else {
            // More central, tightly packed layout
            state.simulation
                .force('x', d3.forceX().strength(0.1).x(GALAXY_CONFIG.width / 2))
                .force('y', d3.forceY().strength(0.1).y(GALAXY_CONFIG.height / 2));
        }
        
        // Restart simulation
        state.simulation.alpha(0.3).restart();
    }

    // Update node sizes
    function updateNodeSizes(sizeValue) {
        const size = parseInt(sizeValue);
        state.nodeSize = size;
        
        // Update node radii
        state.nodes.forEach(node => {
            node.radius = size;
        });
        
        // Update visual elements
        d3.selectAll('.paper-node circle')
            .transition()
            .duration(300)
            .attr('r', d => d.radius);
        
        d3.selectAll('.paper-node .node-glow')
            .transition()
            .duration(300)
            .attr('r', d => d.radius * 1.5);
        
        // Update collision force
        state.simulation
            .force('collision', d3.forceCollide().radius(d => d.radius * 1.5))
            .alpha(0.3)
            .restart();
    }


    // Hide loading screen
    function hideLoading() {
        elements.loadingContainer.style.opacity = 0;
        setTimeout(() => {
            elements.loadingContainer.style.display = 'none';
        }, 500);
    }

    // Add event listeners for all UI controls
    function addEventListeners() {
        // Close paper detail panel
        elements.closePanel.addEventListener('click', () => {
            elements.paperDetail.classList.remove('active');
            d3.selectAll('.paper-node').classed('highlighted', false);
        });
        
        // Close journey guide panel
        elements.closeJourney.addEventListener('click', () => {
            elements.journeyGuide.classList.remove('active');
            state.activeJourney = null;
            state.container.selectAll('.journey-path').remove();
        });

        elements.paperLink.addEventListener('click', (event) => {
            event.preventDefault();
            const link = event.currentTarget.getAttribute("data-paper-link");
            const title = event.currentTarget.getAttribute("data-paper-title");
            if (link) {
                createPaperPopup(link, title);
            }
        });
        
        // Journey navigation
        elements.nextPaper.addEventListener('click', nextJourneyStep);
        elements.prevPaper.addEventListener('click', prevJourneyStep);
        
        // Zoom controls
        elements.zoomIn.addEventListener('click', () => {
            state.svg.transition()
                .duration(300)
                .call(state.zoom.scaleBy, 1.5);
        });
        
        elements.zoomOut.addEventListener('click', () => {
            state.svg.transition()
                .duration(300)
                .call(state.zoom.scaleBy, 0.75);
        });
        
        elements.resetView.addEventListener('click', () => {
            state.svg.transition()
                .duration(500)
                .call(state.zoom.transform, d3.zoomIdentity);
        });
        
        // View mode toggle
        elements.galaxyView.addEventListener('click', () => toggleViewMode('galaxy'));
        elements.clusterView.addEventListener('click', () => toggleViewMode('cluster'));
        
        // Node size slider
        elements.nodeSize.addEventListener('input', (e) => {
            updateNodeSizes(e.target.value);
        });
        
        // Resize handler
        window.addEventListener('resize', debounce(() => {
            // Update dimensions
            const containerRect = elements.galaxyContainer.getBoundingClientRect();
            GALAXY_CONFIG.width = containerRect.width;
            GALAXY_CONFIG.height = containerRect.height;
            
            // Update simulation forces
            updateForceSimulation();
        }, 250));
    }

    // Debounce function for resize events
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }

    // Update force simulation with current dimensions
    function updateForceSimulation() {
        if (!state.simulation) return;
        
        // Update forces
        if (state.viewMode === 'galaxy') {
            state.simulation
                .force('x', d3.forceX().strength(GALAXY_CONFIG.centerForce).x(d => {
                    const clusterData = GALAXY_CONFIG.clusters[d.cluster];
                    return clusterData.x * GALAXY_CONFIG.width;
                }))
                .force('y', d3.forceY().strength(GALAXY_CONFIG.centerForce).y(d => {
                    const clusterData = GALAXY_CONFIG.clusters[d.cluster];
                    return clusterData.y * GALAXY_CONFIG.height;
                }));
        } else {
            state.simulation
                .force('x', d3.forceX().strength(0.1).x(GALAXY_CONFIG.width / 2))
                .force('y', d3.forceY().strength(0.1).y(GALAXY_CONFIG.height / 2));
        }
        
        // Restart simulation
        state.simulation.alpha(0.1).restart();
    }

    function createPaperPopup(pdfUrl, paperTitle) {
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
        
        // Simplify the close button click handler
        closeBtn.onclick = function(event) {
            event.preventDefault();
            event.stopPropagation();
            // Directly remove the overlay from the body
            if (overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
        };
        
        header.appendChild(titleEl);
        header.appendChild(closeBtn);
        
        // 4. Body with the PDF embedded
        const body = document.createElement('div');
        body.className = 'paper-popup-body';
        
        const embed = document.createElement('embed');
        embed.className = 'paper-popup-embed';
        embed.src = pdfUrl;
        embed.type = 'application/pdf';
        embed.style.width = '100%';
        embed.style.height = '100%';
        
        body.appendChild(embed);
        
        // 5. Footer with external link option
        const footer = document.createElement('div');
        footer.className = 'paper-popup-footer';
        
        const externalLink = document.createElement('a');
        externalLink.className = 'paper-popup-external-link';
        externalLink.href = pdfUrl;
        externalLink.target = '_blank';
        externalLink.textContent = 'Open in new tab';
        
        footer.appendChild(externalLink);
        
        // 6. Assemble the popup
        popupContent.appendChild(header);
        popupContent.appendChild(body);
        popupContent.appendChild(footer);
        overlay.appendChild(popupContent);
        
        // Prevent clicks on popup from closing it
        popupContent.onclick = function(event) {
            event.stopPropagation();
        };
        
        // Allow clicking outside popup to close it
        overlay.onclick = function(event) {
            if (event.target === overlay) {
                document.body.removeChild(overlay);
            }
        };
        
        // 7. Add to document
        document.body.appendChild(overlay);
    }
    
    // Function to open paper in popup
    function openPaperLink(paper) {
        if (paper && paper.link) {
            createPaperPopup(paper.link, paper.title);
        }
    }

    // ----- INITIALIZATION -----
    
    // Start by fetching initial data
    fetchInitialData();
});