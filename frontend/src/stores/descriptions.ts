export const possibleDescriptionValues: { [id: string]: string[] } = {
	genre: ['electronic', 'classical', 'soundtrack', 'pop', 'experimental', 'ambient'],
	mood: ['dark', 'melodic', 'film', 'energetic', 'happy', 'relaxing', 'emotional', 'slow', 'epic'],
	instruments: [
		'piano',
		'hammond organ',
		'synth lead',
		'vibraphone',
		'clavinet',
		'acoustic guitar',
		'clarinet',
		'bassoon',
		'trumpet',
		'synth bass',
		'harmonica',
		'ocarina',
		'flute',
		'violin'
	]
};

export const possibleDescriptionValuesEntries = Object.entries(possibleDescriptionValues);
