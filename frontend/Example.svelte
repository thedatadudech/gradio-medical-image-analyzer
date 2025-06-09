<script lang="ts">
	export let value: {
		image?: any;
		analysis?: any;
		report?: string;
	} | null;
	export let type: "gallery" | "table";
	export let selected = false;
</script>

<div
	class="example-container"
	class:table={type === "table"}
	class:gallery={type === "gallery"}
	class:selected
>
	{#if value}
		<div class="example-content">
			{#if value.image}
				<div class="image-preview">
					{#if typeof value.image === 'string'}
						<img src={value.image} alt="Medical scan example" />
					{:else if value.image.url}
						<img src={value.image.url} alt="Medical scan example" />
					{:else}
						<div class="placeholder">📷 Image</div>
					{/if}
				</div>
			{/if}
			
			{#if value.analysis}
				<div class="analysis-preview">
					{#if value.analysis.modality}
						<span class="modality-badge">{value.analysis.modality}</span>
					{/if}
					
					{#if value.analysis.point_analysis?.tissue_type}
						<span class="tissue-type">
							{value.analysis.point_analysis.tissue_type.icon || ''} 
							{value.analysis.point_analysis.tissue_type.type || 'Unknown'}
						</span>
					{/if}
					
					{#if value.analysis.segmentation?.interpretation?.obesity_risk}
						<span class="risk-badge risk-{value.analysis.segmentation.interpretation.obesity_risk}">
							Risk: {value.analysis.segmentation.interpretation.obesity_risk}
						</span>
					{/if}
				</div>
			{/if}
		</div>
	{:else}
		<div class="empty-example">No example</div>
	{/if}
</div>

<style>
	.example-container {
		overflow: hidden;
		border-radius: var(--radius-sm);
		background: var(--background-fill-secondary);
		position: relative;
		transition: all 0.2s ease;
	}
	
	.example-container:hover {
		transform: translateY(-2px);
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
	}
	
	.example-container.selected {
		border: 2px solid var(--color-accent);
	}
	
	.example-container.table {
		display: flex;
		align-items: center;
		padding: 0.5rem;
		gap: 0.5rem;
	}
	
	.example-container.gallery {
		aspect-ratio: 1;
	}
	
	.example-content {
		display: flex;
		flex-direction: column;
		height: 100%;
	}
	
	.table .example-content {
		flex-direction: row;
		align-items: center;
		gap: 0.5rem;
	}
	
	.image-preview {
		flex: 1;
		overflow: hidden;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--background-fill-primary);
	}
	
	.gallery .image-preview {
		height: 70%;
	}
	
	.table .image-preview {
		width: 60px;
		height: 60px;
		flex: 0 0 60px;
		border-radius: var(--radius-sm);
	}
	
	.image-preview img {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}
	
	.placeholder {
		color: var(--body-text-color-subdued);
		font-size: 2rem;
		opacity: 0.5;
	}
	
	.analysis-preview {
		padding: 0.5rem;
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
		align-items: center;
		font-size: 0.875rem;
	}
	
	.gallery .analysis-preview {
		background: var(--background-fill-primary);
		border-top: 1px solid var(--border-color-primary);
	}
	
	.modality-badge {
		background: var(--color-accent);
		color: white;
		padding: 0.125rem 0.5rem;
		border-radius: var(--radius-sm);
		font-weight: bold;
		font-size: 0.75rem;
	}
	
	.tissue-type {
		background: var(--background-fill-secondary);
		padding: 0.125rem 0.5rem;
		border-radius: var(--radius-sm);
		border: 1px solid var(--border-color-primary);
	}
	
	.risk-badge {
		padding: 0.125rem 0.5rem;
		border-radius: var(--radius-sm);
		font-weight: bold;
		font-size: 0.75rem;
	}
	
	.risk-normal {
		background: #d4edda;
		color: #155724;
	}
	
	.risk-moderate {
		background: #fff3cd;
		color: #856404;
	}
	
	.risk-high, .risk-severe {
		background: #f8d7da;
		color: #721c24;
	}
	
	.empty-example {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
		color: var(--body-text-color-subdued);
		font-style: italic;
	}
</style>