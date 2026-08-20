/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.Setter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;

/**
 * This class contains a Lucene's IndexInput with a reader buffer.
 * A Java reference of this class will be passed to native engines, then 'copyBytes' method will be
 * called by native engine via JNI API.
 * Therefore, this class servers as a read layer in native engines to read the bytes it wants.
 */
public class IndexInputWithBuffer {
    private IndexInput indexInput;
    private long contentLength;
    // 64K buffer.
    private byte[] buffer = new byte[64 * 1024];

    /**
     * Lazily supplies the field's full-precision vectors, used only to reconstruct native flat storage when loading a
     * graph-only .faiss produced by FP32 flat-vector deduplication (IO_FLAG_SKIP_STORAGE). Set right before
     * {@code loadIndex}; the supplier is invoked (once) only if native calls {@link #copyVectors(int)}, which happens
     * only for a graph-only index. For normal (non-deduped) loads it is never invoked, so there is no cost or reader
     * access. Null when reconstruction cannot apply (non-FAISS/non-FP32/quantized).
     */
    @Setter
    private CheckedSupplier<KNNVectorValues<?>, IOException> knnVectorValuesSupplier;
    // Materialized (lazily, on first copyVectors call) from knnVectorValuesSupplier.
    private KNNVectorValues<?> knnVectorValues;
    // Reused staging buffer for batched vector streaming to native (sized maxVectors * dimension, lazily).
    private float[] vectorBuffer;
    // Once the vector iterator is exhausted, do not advance it again: calling nextDoc() past NO_MORE_DOCS on a sparse
    // (IndexedDISI-backed) iterator throws an AssertionError. This lets native call copyVectors one final time safely.
    private boolean vectorsExhausted;

    public IndexInputWithBuffer(@NonNull IndexInput indexInput) {
        this.indexInput = indexInput;
        this.contentLength = indexInput.length();
    }

    /**
     * This method will be invoked in native engines via JNI API.
     * Then it will call IndexInput to read required bytes then copy them into a read buffer.
     *
     * @param nbytes Desired number of bytes to be read.
     * @return The number of read bytes in a buffer.
     * @throws IOException
     */
    private int copyBytes(long nbytes) throws IOException {
        final int readBytes = (int) Math.min(nbytes, buffer.length);
        indexInput.readBytes(buffer, 0, readBytes);
        return readBytes;
    }

    private long remainingBytes() {
        return contentLength - indexInput.getFilePointer();
    }

    /**
     * Invoked from native engines via JNI to stream full-precision FP32 vectors when reconstructing the native flat
     * storage for a graph-only .faiss (FP32 flat-vector dedup). Advances {@link #knnVectorValues} and stages up to
     * {@code maxVectors} consecutive vectors (row-major, {@code dimension} floats each) into the reused
     * {@link #vectorBuffer}, which native reads under a JNI critical section. Vectors are yielded in doc-id iteration
     * order, which matches the FAISS ordinal order they were added in at write time.
     *
     * @param maxVectors maximum number of vectors to stage in this batch.
     * @return the number of vectors staged (0 once the iterator is exhausted).
     * @throws IOException on read failure.
     */
    private int copyVectors(int maxVectors) throws IOException {
        if (vectorsExhausted) {
            return 0;
        }
        // Materialize the vector values lazily on first use, so normal (non-graph-only) loads never pay for it.
        if (knnVectorValues == null) {
            if (knnVectorValuesSupplier == null) {
                vectorsExhausted = true;
                return 0;
            }
            knnVectorValues = knnVectorValuesSupplier.get();
        }
        // Only full-precision float vectors are reconstructed this way; native never calls this otherwise.
        if ((knnVectorValues instanceof KNNFloatVectorValues) == false) {
            vectorsExhausted = true;
            return 0;
        }
        final KNNFloatVectorValues floatVectorValues = (KNNFloatVectorValues) knnVectorValues;
        int copied = 0;
        while (copied < maxVectors) {
            if (knnVectorValues.nextDoc() == DocIdSetIterator.NO_MORE_DOCS) {
                vectorsExhausted = true;
                break;
            }
            // getVector() returns a shared reference; copy it into the staging buffer.
            final float[] vector = floatVectorValues.getVector();
            final int dimension = vector.length;
            final int requiredCapacity = maxVectors * dimension;
            if (vectorBuffer == null || vectorBuffer.length < requiredCapacity) {
                vectorBuffer = new float[requiredCapacity];
            }
            System.arraycopy(vector, 0, vectorBuffer, copied * dimension, dimension);
            copied++;
        }
        return copied;
    }

    @Override
    public String toString() {
        return "{indexInput=" + indexInput + ", len(buffer)=" + buffer.length + "}";
    }
}
