<script lang="ts">
	import {
		ListBox,
		ListBoxItem,
		ProgressRadial,
		getToastStore,
		type ToastSettings
	} from '@skeletonlabs/skeleton';
	import abcjs from 'abcjs';
	import { onMount } from 'svelte';
	import { possibleDescriptionValuesEntries } from '../stores/descriptions';
	import { PUBLIC_MIDISTRAL_SERVER } from '$env/static/public';

	onMount(() => {});

	let loading: boolean = false;
	let fileUuid: string | null = null;
	let abcNotationId: string | null = null;
	async function generate() {
		loading = true;
		feedbackLiked = false;
		fileUuid = null;
		try {
			let response = await fetch(`${PUBLIC_MIDISTRAL_SERVER}/generate`, {
				method: 'post',
				headers: {
					Accept: 'application/json',
					'Content-Type': 'application/json'
				},

				//make sure to serialize your JSON body
				body: JSON.stringify(selectedValues)
			});
			let responseJson = await response.json();

			if (!response.ok) {
				let errorMessage = 'Generation failed';
				if (responseJson.error) {
					errorMessage = responseJson.error;
				} else if (responseJson.detail) {
					errorMessage = responseJson.detail;
				}
				const t: ToastSettings = {
					message: errorMessage,
					background: 'variant-soft-warning',
					hideDismiss: true
				};
				toastStore.trigger(t);
			} else {
				let abcnotation = responseJson['abc_notation'];
				fileUuid = responseJson['file_uuid'];
				abcNotationId = responseJson['id'];
				loading = false;

				setTimeout(function () {
					abcjs.renderAbc('paper', abcnotation, {
						responsive: 'resize',
						selectTypes: false,
						wrap: { minSpacing: 1.8, maxSpacing: 2.7, preferredMeasuresPerLine: 4 },
						staffwidth: 600
					});
				}, 50);
			}
		} catch (error) {}
		loading = false;
	}

	// Input Chip
	let selectedValues: { [id: string]: string[] } = {
		genre: [],
		mood: [],
		instruments: ['piano']
	};

	$: selectedValues && selectedValuesChanged();

	const selectedValuesChanged = () => {
		// Fix max number of selected items to 1 for genre and instruments
		if (selectedValues.genre.length > 1) {
			selectedValues.genre = selectedValues.genre.slice(1);
		}
		if (selectedValues.instruments.length > 1) {
			selectedValues.instruments = selectedValues.instruments.slice(1);
		}
		if (selectedValues.mood.length > 2) {
			selectedValues.mood = selectedValues.mood.slice(1);
		}
	};

	let feedbackLiked = false;
	const toastStore = getToastStore();

	async function sendFeedback(abcAnnotationId: string): Promise<void> {
		feedbackLiked = !feedbackLiked;
		let response = await fetch(
			`${PUBLIC_MIDISTRAL_SERVER}/feedback?id=${abcAnnotationId}&liked=${feedbackLiked}`,
			{
				method: 'post',
				headers: {
					Accept: 'application/json',
					'Content-Type': 'application/json'
				}
			}
		);
		let responseJson = await response.json();

		if (!response.ok) {
			let errorMessage = 'Generation failed';
			if (responseJson.error) {
				errorMessage = responseJson.error;
			} else if (responseJson.detail) {
				errorMessage = responseJson.detail;
			}
			feedbackLiked = !feedbackLiked;
			const t: ToastSettings = {
				message: errorMessage,
				background: 'variant-soft-warning',
				hideDismiss: true
			};
			toastStore.trigger(t);
		} else {
			const t: ToastSettings = {
				message: `Thank you for your feedback !`,
				background: 'variant-filled-primary',
				hideDismiss: true
			};
			toastStore.trigger(t);
		}
	}

	async function download(dataurl: string, fileName: string) {
		const response = await fetch(dataurl);
		const blob = await response.blob();

		const link = document.createElement('a');
		link.href = URL.createObjectURL(blob);
		link.download = fileName;
		link.click();
	}
</script>

<div class="container h-full mx-auto flex justify-center">
	<div class="space-y-8 flex flex-col items-center">
		<p class="text-2xl pt-8">Create your own unique MIDI music</p>
		<div class="grid grid-cols-3 sm:gap-2 md:gap-4 pt-8">
			{#each possibleDescriptionValuesEntries as [key, values]}
				<div class="flex flex-col">
					<h4 class="h4 pb-4 text-center">{key}</h4>
					<ListBox
						class="max-h-[200px] overflow-y-auto text-center card p-2"
						active="variant-filled-surface"
						hover="hover:variant-soft-primary"
						multiple
					>
						{#each values as v}
							<ListBoxItem bind:group={selectedValues[key]} name="medium" value={v}>{v}</ListBoxItem
							>
						{/each}
					</ListBox>
				</div>
			{/each}
		</div>

		<div class="flex w-fit justify-center pb-4">
			<button
				on:click={async () => {
					await generate();
				}}
				class="btn variant-filled-primary"
			>
				Generate
			</button>
		</div>
		{#if loading}
			<div class="flex w-8 justify-center">
				<ProgressRadial />
			</div>
		{:else if fileUuid != null && abcNotationId != null}
			<div class="flex items-center gap-4 px-8">
				<audio controls>
					<source src={`${PUBLIC_MIDISTRAL_SERVER}/file/${fileUuid}.ogg`} type="audio/ogg" />
					Your browser does not support the audio element.
				</audio>
				<div class="flex flex-col gap-2">
					<button
						class="chip {feedbackLiked ? 'variant-filled-primary' : 'variant-soft'}"
						on:click={async () => {
							if (abcNotationId != null) {
								await sendFeedback(abcNotationId);
							}
						}}
						on:keypress
					>
						<i class="fa-solid fa-heart"></i>
						<span>Like</span>
					</button>
					<button
						class="chip variant-soft"
						on:click={() =>
							download(`${PUBLIC_MIDISTRAL_SERVER}/file/${fileUuid}.ogg`, 'audio.ogg')}
					>
						<i class="fa-solid fa-file-waveform"></i>
						<span>OGG file</span>
					</button>
					<button
						class="chip variant-soft"
						on:click={() =>
							download(`${PUBLIC_MIDISTRAL_SERVER}/file/${fileUuid}.midi`, 'audio.midi')}
					>
						<i class="fa-solid fa-wave-square"></i>
						<span>MIDI file</span>
					</button>
				</div>
			</div>
			<div class="flex w-full px-8 pb-8">
				<div id="paper"></div>
			</div>
		{/if}
	</div>
</div>

<style lang="postcss">
</style>
