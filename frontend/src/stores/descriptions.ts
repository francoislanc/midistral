export const possibleDescriptionValues: { [id: string]: string[] } = {
	genre: ['classical', 'electronic', 'pop', 'soundtrack'],
	mood: ['positive', 'energetic', 'calm', 'emotional', 'film'],
	instruments: [
		'acoustic guitar',
		'piano',
		'clarinet',
		'ocarina',
		'synth lead',
		'trombone',
		'trumpet'
	]
};

export const possibleDescriptionValuesEntries = Object.entries(possibleDescriptionValues);
