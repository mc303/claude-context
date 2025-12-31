import { pipeline, type Pipeline } from '@xenova/transformers';
import { Embedding, EmbeddingVector } from './base-embedding';

interface ProgressCallbackArgs {
    file: string;
    progress: number;
    status: string;
}

export interface MiniLMEmbeddingConfig {
    model?: string;
    progressCallback?: (progress: ProgressCallbackArgs) => void;
}

/**
 * CPU-based embedding using all-MiniLM-L6-v2 model
 * Runs locally using @xenova/transformers (ONNX runtime)
 */
export class MiniLMEmbedding extends Embedding {
    private static INSTANCE: MiniLMEmbedding | null = null;
    private pipeline: any = null;
    private config: MiniLMEmbeddingConfig;
    private dimension: number = 384; // all-MiniLM-L6-v2 produces 384-dimensional vectors
    protected maxTokens: number = 512; // MiniLM has lower token limits than OpenAI models
    private modelId: string;

    constructor(config: MiniLMEmbeddingConfig = {}) {
        super();
        this.config = config;
        // Default model: all-MiniLM-L6-v2 (384 dimensions, fast and efficient)
        this.modelId = config.model || 'Xenova/all-MiniLM-L6-v2';
    }

    /**
     * Initialize the embedding pipeline (lazy loading)
     */
    private async initPipeline(): Promise<any> {
        if (this.pipeline) {
            return this.pipeline;
        }

        console.log(`[MiniLM] Loading model ${this.modelId}...`);
        console.log('[MiniLM] First run will download the model (~100MB). Subsequent runs will use cached model.');

        this.pipeline = await pipeline('feature-extraction', this.modelId, {
            progress_callback: this.config.progressCallback || ((progress: ProgressCallbackArgs) => {
                if (progress.status === 'downloading') {
                    console.log(`[MiniLM] Downloading ${progress.file}: ${Math.round(progress.progress)}%`);
                } else if (progress.status === 'loading') {
                    console.log(`[MiniLM] Loading ${progress.file}...`);
                }
            })
        });

        console.log('[MiniLM] Model loaded successfully');
        return this.pipeline;
    }

    async detectDimension(testText: string = "test"): Promise<number> {
        return this.dimension;
    }

    async embed(text: string): Promise<EmbeddingVector> {
        const processedText = this.preprocessText(text);
        const pipe = await this.initPipeline();

        // Generate embedding
        const output = await pipe(processedText, {
            pooling: 'mean',
            normalize: true
        });

        // Convert Tensor to array
        const vector = Array.from(output.data as Float32Array);

        return {
            vector,
            dimension: vector.length
        };
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        const processedTexts = this.preprocessTexts(texts);
        const pipe = await this.initPipeline();

        const results: EmbeddingVector[] = [];

        // Process texts sequentially (the ONNX runtime handles batching internally)
        for (const text of processedTexts) {
            const output = await pipe(text, {
                pooling: 'mean',
                normalize: true
            });

            const vector = Array.from(output.data as Float32Array);
            results.push({
                vector,
                dimension: vector.length
            });
        }

        return results;
    }

    getDimension(): number {
        return this.dimension;
    }

    getProvider(): string {
        return 'MiniLM';
    }

    /**
     * Get singleton instance of MiniLM embedding
     */
    static async getInstance(config?: MiniLMEmbeddingConfig): Promise<MiniLMEmbedding> {
        if (!MiniLMEmbedding.INSTANCE) {
            MiniLMEmbedding.INSTANCE = new MiniLMEmbedding(config);
        }
        return MiniLMEmbedding.INSTANCE;
    }

    /**
     * Reset singleton instance (useful for testing or model switching)
     */
    static resetInstance(): void {
        MiniLMEmbedding.INSTANCE = null;
    }

    /**
     * Get list of supported MiniLM models
     */
    static getSupportedModels(): Record<string, { dimension: number; description: string }> {
        return {
            'Xenova/all-MiniLM-L6-v2': {
                dimension: 384,
                description: 'Fast and efficient CPU-based embedding model (recommended)'
            },
            'Xenova/all-MiniLM-L12-v2': {
                dimension: 384,
                description: 'More accurate but slower variant with 12 layers'
            },
            'Xenova/all-mpnet-base-v2': {
                dimension: 768,
                description: 'Higher quality but larger and slower (768 dimensions)'
            }
        };
    }

    /**
     * Clean up resources
     */
    async dispose(): Promise<void> {
        this.pipeline = null;
    }
}
