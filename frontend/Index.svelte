<script lang="ts">
	import { Block, BlockLabel, Empty, IconButton, Upload, UploadText } from "@gradio/atoms";
	import { Image } from "@gradio/icons";
	import { StatusTracker } from "@gradio/statustracker";
	import type { LoadingStatus } from "@gradio/statustracker";
	import { _ } from "svelte-i18n";
	import { tick } from "svelte";
	import { Upload as UploadIcon } from "@gradio/icons";
	
	export let elem_id = "";
	export let elem_classes: string[] = [];
	export let visible = true;
	export let value: {
		image?: any;
		analysis?: any;
		report?: string;
	} | null = null;
	export let label: string;
	export let show_label: boolean;
	export let show_download_button: boolean;
	export let root: string;
	export let proxy_url: null | string;
	export let loading_status: LoadingStatus;
	export let container = true;
	export let scale: number | null = null;
	export let min_width: number | undefined = undefined;
	export let gradio: any;
	
	// Analysis parameters
	export let analysis_mode: "structured" | "visual" = "structured";
	export let include_confidence = true;
	export let include_reasoning = true;
	export let modality: "CT" | "CR" | "DX" | "RX" | "DR" = "CT";
	export let task: "analyze_point" | "segment_fat" | "full_analysis" = "full_analysis";
	
	let dragging = false;
	let pending_upload = false;
	let uploaded_file: File | null = null;
	let roi = { x: 256, y: 256, radius: 10 };
	let show_roi = false;
	let analysis_results: any = null;
	let visual_report = "";
	
	$: value = {
		image: uploaded_file,
		analysis: analysis_results,
		report: visual_report
	};
	
	// DICOM and image loading
	async function load_file(file: File) {
		const file_url = URL.createObjectURL(file);
		const file_ext = file.name.split('.').pop()?.toLowerCase() || '';
		
		try {
			// Always try DICOM first for files without extensions or with DICOM extensions
			// This matches the backend behavior with force=True
			if (!file_ext || file_ext === 'dcm' || file_ext === 'dicom' || 
				file.type === 'application/dicom' || file.name.startsWith('IM_')) {
				// For DICOM, we need server-side processing
				// Send to backend for processing
				const formData = new FormData();
				formData.append('file', file);
				
				// This would call the backend to process DICOM
				const response = await fetch(`${root}/process_dicom`, {
					method: 'POST',
					body: formData
				});
				
				if (response.ok) {
					const data = await response.json();
					return data;
				}
			}
			
			// Fallback to regular image handling
			return {
				url: file_url,
				name: file.name,
				size: file.size,
				type: file.type || 'application/octet-stream'
			};
		} catch (error) {
			console.error("Error loading file:", error);
			throw error;
		}
	}
	
	function handle_upload({ detail }: CustomEvent<File>) {
		pending_upload = true;
		const file = detail;
		
		load_file(file).then((data) => {
			uploaded_file = file;
			pending_upload = false;
			
			// Trigger analysis
			if (gradio.dispatch) {
				gradio.dispatch("upload", {
					file: file,
					data: data
				});
			}
		}).catch((error) => {
			console.error("Upload error:", error);
			pending_upload = false;
		});
	}
	
	function handle_clear() {
		value = null;
		uploaded_file = null;
		analysis_results = null;
		visual_report = "";
		gradio.dispatch("clear");
	}
	
	function handle_roi_click(event: MouseEvent) {
		if (!show_roi) return;
		
		const rect = (event.target as HTMLElement).getBoundingClientRect();
		roi.x = Math.round(event.clientX - rect.left);
		roi.y = Math.round(event.clientY - rect.top);
		
		// Use change event for ROI updates
		if (gradio.dispatch) {
			gradio.dispatch("change", { roi });
		}
	}
	
	function create_visual_report(results: any) {
		if (!results) return "";
		
		let html = `<div class="medical-report">`;
		html += `<h3>🏥 Medical Image Analysis Report</h3>`;
		
		// Basic info
		html += `<div class="report-section">`;
		html += `<h4>📋 Basic Information</h4>`;
		html += `<p><strong>Modality:</strong> ${results.modality || 'Unknown'}</p>`;
		html += `<p><strong>Timestamp:</strong> ${results.timestamp || 'N/A'}</p>`;
		html += `</div>`;
		
		// Point analysis
		if (results.point_analysis) {
			const pa = results.point_analysis;
			html += `<div class="report-section">`;
			html += `<h4>🎯 Point Analysis</h4>`;
			html += `<p><strong>Location:</strong> (${pa.location?.x}, ${pa.location?.y})</p>`;
			
			if (results.modality === 'CT') {
				html += `<p><strong>HU Value:</strong> ${pa.hu_value?.toFixed(1) || 'N/A'}</p>`;
			} else {
				html += `<p><strong>Intensity:</strong> ${pa.intensity?.toFixed(3) || 'N/A'}</p>`;
			}
			
			if (pa.tissue_type) {
				html += `<p><strong>Tissue Type:</strong> ${pa.tissue_type.icon || ''} ${pa.tissue_type.type || 'Unknown'}</p>`;
			}
			
			if (include_confidence && pa.confidence !== undefined) {
				html += `<p><strong>Confidence:</strong> ${pa.confidence}</p>`;
			}
			
			if (include_reasoning && pa.reasoning) {
				html += `<p class="reasoning">💭 ${pa.reasoning}</p>`;
			}
			
			html += `</div>`;
		}
		
		// Segmentation results
		if (results.segmentation?.statistics) {
			const stats = results.segmentation.statistics;
			
			if (results.modality === 'CT' && stats.total_fat_percentage !== undefined) {
				html += `<div class="report-section">`;
				html += `<h4>🔬 Fat Segmentation</h4>`;
				html += `<div class="stats-grid">`;
				html += `<div><strong>Total Fat:</strong> ${stats.total_fat_percentage.toFixed(1)}%</div>`;
				html += `<div><strong>Subcutaneous:</strong> ${stats.subcutaneous_fat_percentage.toFixed(1)}%</div>`;
				html += `<div><strong>Visceral:</strong> ${stats.visceral_fat_percentage.toFixed(1)}%</div>`;
				html += `<div><strong>V/S Ratio:</strong> ${stats.visceral_subcutaneous_ratio.toFixed(2)}</div>`;
				html += `</div>`;
				
				if (results.segmentation.interpretation) {
					const interp = results.segmentation.interpretation;
					html += `<div class="interpretation">`;
					html += `<p><strong>Obesity Risk:</strong> <span class="risk-${interp.obesity_risk}">${interp.obesity_risk.toUpperCase()}</span></p>`;
					html += `<p><strong>Visceral Risk:</strong> <span class="risk-${interp.visceral_risk}">${interp.visceral_risk.toUpperCase()}</span></p>`;
					
					if (interp.recommendations?.length > 0) {
						html += `<p><strong>Recommendations:</strong></p>`;
						html += `<ul>`;
						interp.recommendations.forEach((rec: string) => {
							html += `<li>${rec}</li>`;
						});
						html += `</ul>`;
					}
					html += `</div>`;
				}
				html += `</div>`;
			} else if (results.segmentation.tissue_distribution) {
				html += `<div class="report-section">`;
				html += `<h4>🦴 Tissue Distribution</h4>`;
				html += `<div class="tissue-grid">`;
				
				const tissues = results.segmentation.tissue_distribution;
				const icons: Record<string, string> = {
					bone: '🦴',
					soft_tissue: '🔴',
					air: '🌫️',
					metal: '⚙️',
					fat: '🟡',
					fluid: '💧'
				};
				
				Object.entries(tissues).forEach(([tissue, percentage]) => {
					if (percentage as number > 0) {
						html += `<div class="tissue-item">`;
						html += `<div class="tissue-icon">${icons[tissue] || '📍'}</div>`;
						html += `<div class="tissue-name">${tissue.replace('_', ' ')}</div>`;
						html += `<div class="tissue-percentage">${(percentage as number).toFixed(1)}%</div>`;
						html += `</div>`;
					}
				});
				
				html += `</div>`;
				
				if (results.segmentation.clinical_findings?.length > 0) {
					html += `<div class="clinical-findings">`;
					html += `<p><strong>⚠️ Clinical Findings:</strong></p>`;
					html += `<ul>`;
					results.segmentation.clinical_findings.forEach((finding: any) => {
						html += `<li>${finding.description} (Confidence: ${finding.confidence})</li>`;
					});
					html += `</ul>`;
					html += `</div>`;
				}
				
				html += `</div>`;
			}
		}
		
		// Quality metrics
		if (results.quality_metrics) {
			const quality = results.quality_metrics;
			html += `<div class="report-section">`;
			html += `<h4>📊 Image Quality</h4>`;
			html += `<p><strong>Overall Quality:</strong> <span class="quality-${quality.overall_quality}">${quality.overall_quality?.toUpperCase() || 'UNKNOWN'}</span></p>`;
			
			if (quality.issues?.length > 0) {
				html += `<p><strong>Issues:</strong> ${quality.issues.join(', ')}</p>`;
			}
			
			html += `</div>`;
		}
		
		html += `</div>`;
		
		return html;
	}
	
	// Update visual report when analysis changes
	$: if (analysis_results) {
		visual_report = create_visual_report(analysis_results);
	}
</script>

<Block
	{visible}
	{elem_id}
	{elem_classes}
	{container}
	{scale}
	{min_width}
	allow_overflow={false}
	padding={true}
>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
	/>
	
	<BlockLabel
		{show_label}
		Icon={Image}
		label={label || "Medical Image Analyzer"}
	/>
	
	{#if value === null || !uploaded_file}
		<Upload
			on:load={handle_upload}
			filetype="*"
			{root}
			{dragging}
		>
			<UploadText i18n={gradio.i18n} type="file">
			Drop Medical Image File Here - or - Click to Upload<br/>
			<span style="font-size: 0.9em; color: var(--body-text-color-subdued);">
				Supports: DICOM (.dcm), Images (.png, .jpg), and files without extensions (IM_0001, etc.)
			</span>
		</UploadText>
		</Upload>
	{:else}
		<div class="analyzer-container">
			<div class="controls">
				<IconButton Icon={UploadIcon} on:click={handle_clear} />
				
				<select bind:value={modality} class="modality-select">
					<option value="CT">CT</option>
					<option value="CR">CR (X-Ray)</option>
					<option value="DX">DX (X-Ray)</option>
					<option value="RX">RX (X-Ray)</option>
					<option value="DR">DR (X-Ray)</option>
				</select>
				
				<select bind:value={task} class="task-select">
					<option value="analyze_point">Point Analysis</option>
					<option value="segment_fat">Fat Segmentation (CT)</option>
					<option value="full_analysis">Full Analysis</option>
				</select>
				
				<label class="roi-toggle">
					<input type="checkbox" bind:checked={show_roi} />
					Show ROI
				</label>
			</div>
			
			<div class="image-container" on:click={handle_roi_click}>
				{#if uploaded_file}
					<img src={URL.createObjectURL(uploaded_file)} alt="Medical scan" />
					
					{#if show_roi}
						<div 
							class="roi-marker" 
							style="left: {roi.x}px; top: {roi.y}px; width: {roi.radius * 2}px; height: {roi.radius * 2}px;"
						/>
					{/if}
				{/if}
			</div>
			
			{#if visual_report}
				<div class="report-container">
					{@html visual_report}
				</div>
			{/if}
			
			{#if analysis_mode === "structured" && analysis_results}
				<details class="json-output">
					<summary>JSON Output (for AI Agents)</summary>
					<pre>{JSON.stringify(analysis_results, null, 2)}</pre>
				</details>
			{/if}
		</div>
	{/if}
</Block>

<style>
	.analyzer-container {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
	
	.controls {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		flex-wrap: wrap;
	}
	
	.modality-select, .task-select {
		padding: 0.5rem;
		border: 1px solid var(--border-color-primary);
		border-radius: var(--radius-sm);
		background: var(--background-fill-primary);
	}
	
	.roi-toggle {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		cursor: pointer;
	}
	
	.image-container {
		position: relative;
		overflow: hidden;
		border: 1px solid var(--border-color-primary);
		border-radius: var(--radius-sm);
		cursor: crosshair;
	}
	
	.image-container img {
		width: 100%;
		height: auto;
		display: block;
	}
	
	.roi-marker {
		position: absolute;
		border: 2px solid #ff0000;
		border-radius: 50%;
		pointer-events: none;
		transform: translate(-50%, -50%);
		box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.5);
	}
	
	.report-container {
		background: var(--background-fill-secondary);
		border: 1px solid var(--border-color-primary);
		border-radius: var(--radius-sm);
		padding: 1rem;
		overflow-x: auto;
	}
	
	:global(.medical-report) {
		font-family: var(--font);
		color: var(--body-text-color);
	}
	
	:global(.medical-report h3) {
		color: var(--body-text-color);
		border-bottom: 2px solid var(--color-accent);
		padding-bottom: 0.5rem;
		margin-bottom: 1rem;
	}
	
	:global(.medical-report h4) {
		color: var(--body-text-color);
		margin-top: 1rem;
		margin-bottom: 0.5rem;
	}
	
	:global(.report-section) {
		background: var(--background-fill-primary);
		padding: 1rem;
		border-radius: var(--radius-sm);
		margin-bottom: 1rem;
	}
	
	:global(.stats-grid), :global(.tissue-grid) {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
		gap: 0.5rem;
		margin-top: 0.5rem;
	}
	
	:global(.tissue-item) {
		text-align: center;
		padding: 0.5rem;
		background: var(--background-fill-secondary);
		border-radius: var(--radius-sm);
	}
	
	:global(.tissue-icon) {
		font-size: 2rem;
		margin-bottom: 0.25rem;
	}
	
	:global(.tissue-name) {
		font-weight: bold;
		text-transform: capitalize;
	}
	
	:global(.tissue-percentage) {
		color: var(--color-accent);
		font-size: 1.2rem;
		font-weight: bold;
	}
	
	:global(.reasoning) {
		font-style: italic;
		color: var(--body-text-color-subdued);
		margin-top: 0.5rem;
	}
	
	:global(.interpretation) {
		margin-top: 1rem;
		padding: 0.5rem;
		background: var(--background-fill-secondary);
		border-radius: var(--radius-sm);
	}
	
	:global(.risk-normal) { color: #27ae60; }
	:global(.risk-moderate) { color: #f39c12; }
	:global(.risk-high), :global(.risk-severe) { color: #e74c3c; }
	
	:global(.quality-excellent), :global(.quality-good) { color: #27ae60; }
	:global(.quality-fair) { color: #f39c12; }
	:global(.quality-poor) { color: #e74c3c; }
	
	:global(.clinical-findings) {
		margin-top: 1rem;
		padding: 0.5rem;
		background: #fff3cd;
		border-left: 4px solid #ffc107;
		border-radius: var(--radius-sm);
	}
	
	.json-output {
		margin-top: 1rem;
		background: var(--background-fill-secondary);
		border: 1px solid var(--border-color-primary);
		border-radius: var(--radius-sm);
		padding: 1rem;
	}
	
	.json-output summary {
		cursor: pointer;
		font-weight: bold;
		margin-bottom: 0.5rem;
	}
	
	.json-output pre {
		margin: 0;
		overflow-x: auto;
		font-size: 0.875rem;
		background: var(--background-fill-primary);
		padding: 0.5rem;
		border-radius: var(--radius-sm);
	}
</style>