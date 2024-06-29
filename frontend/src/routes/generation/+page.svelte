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
	import { possibleDescriptionValuesEntries } from '../../stores/descriptions';
	import { PUBLIC_MIDISTRAL_SERVER } from '$env/static/public';

	let characters = 0;
	let myInterval = 0;
	onMount(() => {
		setTimeout(startCompletion, 5000);
	});

	function startCompletion() {
		myInterval = setInterval(incrementCharacters, 50);
	}

	function incrementCharacters() {
		characters += 1;

		if (characters == 1000) {
			clearInterval(myInterval);
		}
	}

	let abcNotations = [
		'X: 1\nM: 4/4\nL: 1/8\nQ:1/4=180\nK:C\nV:1\n%%MIDI program 0\ngc cg cc gc| ^gc d^A Af A=A| f^A ^fA gc cg| cc gc ^gc f^A|\n^Af AA fA ^fA|\n',
		'X: 1\nM: 4/4\nL: 1/8\nQ:1/4=150\nK:C\nV:1\n%%MIDI program 0\nA,z A,z A,z A,>B,| Cz Cz C3/2z/2 A,2| Fz Fz Fz F>D| E/2zE/2 E/2zE/2 E2 B,z|\nA,z A,z A,z A,>B,| Cz Cz C3/2z/2 A,2| Fz Fz Fz F>D| E/2zE/2 E/2zE/2 E2 A\n',
		'X: 1\nM: 4/4\nL: 1/8\nQ:1/4=140\nK:B\nV:1\n%%MIDI program 0\nG/2G/2d/2z/2 G/2B/2z/2G/2 c/2z/2G/2A/2 z/2F/2B/2z/2| F/2G/2z/2D/2 F/2z/2D/2F/2 z/2D/2F/2z/2 E/2G/2z/2E/2| G/2z/2E/2G/2 \n',
		'X: 1\nM: 6/8\nL: 1/8\nQ:1/4=120\nK:C\nV:1\nA/2B/2c2 ccd| edB GG2| Bc2 AAB| AG2 EE2|\nBc2 ccd| edB GG2| BcB ABA| ^G2<A2A2|\nA/2B/2c2 ccd| edB GG2| Bc2 AAB| AG2 EE2|\nBc2 ccd| edB GG2| BcB ABA| ^G2<A2A2|\nf/2e/2g2 gg^f| edB GG2| ga2 aab| ag2 ee2|\nfga gg^f| edB GGA| BcB ABA| ^G2<A2A2|\nf/2e/2g2 gg^f| edB GG2| ga2 aab| ag2 ee2|\nfga gg^f| edB GGA| BcB ABA| ^G2<A2A2|\n',
		'X: 1\nM: 4/4\nL: 1/8\nQ:1/4=220\nK:Db\nV:1\ne4 d2 dd| cB AB2e2e| ee ee dd dd| cB AE2\n',
		'X: 1\nM: 4/4\nL: 1/8\nQ:1/4=200\nK:D\nV:1\nA2 B2 c2 A2| E4 z4| z6 E2| A2 B2 c2 B2-|\nB2 z6| z2 F2 B2 c2| d2 c2 B4| z8|\nz2 E2 c2 e2| cB A2 FE3| z8| E2 f2 f2 f2|\ne2 d4 z2| z6 AA| d2 e2 f2 e2-| e2 z6|\nz2 A2 e2 f2| g2 f2 e4| z8| A2 f2 a2 fe|\nd2 B2<A2 \n'
		// 'X: 1\nM: 4/4\nL: 1/8\nQ:1/4=150\nK:C\nV:1\nF>F, F>F, F>F, F>F,| G>F, G>F, ^G>F, G>F,| E>E, E>E, E>E, E>E,| ^G>E, G>E, =G>E, G>E,|\n',
		// 'X: 1\nM: 4/4\nL: 1/8\nQ:1/4=140\nK:C\nV:1\nEz/2A,z/2E z/2A,z/2 Fz/2F/2-| F/2z/2E z/2Ez/2 Ez/2Ez/2E| Ez/2G,z/2E z/2G,z/2 Ez/2G,/2-| G,/2z/2E z/2G,z/2 Fz/2Fz/2F|\nEz/2Ez/2C z/2Cz/2 Fz/2F/2-| F/2z/2E z/2Ez/2 Ez/2Ez/2E| Ez/2Cz/2E z/2Cz/2 Ez/2C/2-| C/2z/2B, z/2Ez/2 B,z/2B,z/2B,-|\nB,8-|B,8-|B,8-|B,8-|\nB,8-|B,8-|B,8-|B,6 \n'
	];
</script>

<div class="h-full flex justify-center">
	<div class="grid grid-cols-3 sm:gap-2 md:gap-4 pt-8">
		{#each abcNotations as abcNotation, i}
			<div class="card p-4 h-64 w-72 overflow-y-auto">
				<div class="whitespace-pre-wrap">{abcNotation.substring(0, characters)}</div>
			</div>
		{/each}
	</div>
</div>

<style lang="postcss">
</style>
