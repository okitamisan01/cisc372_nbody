#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void computeAccels_shared(
    vector3 *dPos,    // 引数① GPU側の位置配列（読み取り専用）
    double  *dMass,   // 引数② GPU側の質量配列（読み取り専用）
    vector3 *dAccels  // 引数③ GPU側の加速度行列（書き込み先）
) {
    // ── 共有メモリ宣言 ────────────────────────────────
    __shared__ vector3 shPos_i[16];
    // 役割 : このブロックが担当するi側（行）の位置キャッシュ
    // サイズ: 16 = blockDim.x
    // 誰が書く: threadIdx.y == 0 のスレッドだけ
    // 誰が読む: ブロック内の全スレッド

    __shared__ vector3 shPos_j[16];
    // 役割 : このブロックが担当するj側（列）の位置キャッシュ
    // サイズ: 16 = blockDim.y
    // 誰が書く: threadIdx.x == 0 のスレッドだけ
    // 誰が読む: ブロック内の全スレッド

    __shared__ double shMass_j[16];
    // 役割 : このブロックが担当するj側（列）の質量キャッシュ
    // サイズ: 16 = blockDim.y
    // 誰が書く: threadIdx.x == 0 のスレッドだけ
    // 誰が読む: ブロック内の全スレッド

    // ── インデックス計算 ──────────────────────────────
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 物体iのインデックス（行方向）

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // 物体jのインデックス（列方向）

    // ── 共有メモリへの読み込み ────────────────────────
    if (threadIdx.y == 0 && i < NUMENTITIES)
        for(int k=0;k<3;k++) shPos_i[threadIdx.x][k] = dPos[i][k];
    // y方向先頭スレッドだけ書き込む
    // 例: ブロック内の(0,0)(1,0)(2,0)...が担当
    //     → shPos_i[0]=dPos[i0], shPos_i[1]=dPos[i1]...

    if (threadIdx.x == 0 && j < NUMENTITIES) {
        for(int k=0;k<3;k++) shPos_j[threadIdx.y][k] = dPos[j][k];
        shMass_j[threadIdx.y] = dMass[j];
    }
    // x方向先頭スレッドだけ書き込む
    // 例: ブロック内の(0,0)(0,1)(0,2)...が担当
    //     → shPos_j[0]=dPos[j0], shPos_j[1]=dPos[j1]...

    __syncthreads();
    // ↑ここで全スレッドの書き込みが終わるまで待つ
    // これがないと未書き込みデータを読む危険がある！

    if (i >= NUMENTITIES || j >= NUMENTITIES) return;

    // ── 計算（共有メモリから読む）────────────────────
    if (i == j) {
        FILL_VECTOR(dAccels[i*NUMENTITIES+j], 0, 0, 0);

    } else {
        vector3 distance;
        for (int k = 0; k < 3; k++)
            distance[k] = shPos_i[threadIdx.x][k]  // ← 共有メモリ！
                        - shPos_j[threadIdx.y][k];  // ← 共有メモリ！
        //                ^^^^^^^^^^^^^^^^^^^^^^^^
        //                グローバルメモリの代わりに共有メモリから読む
        //                → 何百倍も速い

        double magnitude_sq = distance[0]*distance[0]
                            + distance[1]*distance[1]
                            + distance[2]*distance[2];
        double magnitude  = sqrt(magnitude_sq);
        double accelmag   = -1 * GRAV_CONSTANT
                          * shMass_j[threadIdx.y]   // ← 共有メモリ！
                          / magnitude_sq;

        FILL_VECTOR(dAccels[i*NUMENTITIES+j],
            accelmag * distance[0] / magnitude,
            accelmag * distance[1] / magnitude,
            accelmag * distance[2] / magnitude);
    }
}

// __global__ void computeAccels(
//     vector3 *dPos,    // 引数① GPU側の位置配列（読み取り専用）
//     double  *dMass,   // 引数② GPU側の質量配列（読み取り専用）
//     vector3 *dAccels  // 引数③ GPU側の加速度行列（書き込み先）
// ) {
//     // ── インデックス計算 ──────────────────────────
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     //       ↑ブロック番号  ↑ブロックサイズ  ↑ブロック内番号
//     //       → 物体iのインデックス（行方向）

//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     //       同上（列方向）
//     //       → 物体jのインデックス

//     // 範囲外スレッドは何もしない
//     if (i >= NUMENTITIES || j >= NUMENTITIES) return;

//     // ── 計算 ──────────────────────────────────────
//     if (i == j) {
//         // 自分自身への影響はゼロ
//         FILL_VECTOR(dAccels[i*NUMENTITIES+j], 0, 0, 0);

//     } else {
//         vector3 distance;
//         // distance[k] : iとjの位置の差（方向ベクトル）
//         // distance[0] = x方向の差
//         // distance[1] = y方向の差
//         // distance[2] = z方向の差
//         for (int k = 0; k < 3; k++)
//             distance[k] = dPos[i][k] - dPos[j][k];

//         double magnitude_sq;
//         // 距離の二乗 = x差² + y差² + z差²
//         magnitude_sq = distance[0]*distance[0]
//                      + distance[1]*distance[1]
//                      + distance[2]*distance[2];

//         double magnitude;
//         // 距離（magnitude_sqの平方根）
//         magnitude = sqrt(magnitude_sq);

//         double accelmag;
//         // 加速度の大きさ = -G × 質量j ÷ 距離²
//         // マイナス = 引力（引き寄せられる方向）
//         accelmag = -1 * GRAV_CONSTANT * dMass[j] / magnitude_sq;

//         // 加速度ベクトルを方向ベクトルに大きさをかけて計算
//         FILL_VECTOR(dAccels[i*NUMENTITIES+j],
//             accelmag * distance[0] / magnitude,  // x成分
//             accelmag * distance[1] / magnitude,  // y成分
//             accelmag * distance[2] / magnitude   // z成分
//         );
//     }
// }

__global__ void updateBodies_reduction(
    vector3 *dPos,    // 引数① GPU側の位置配列（読み書き）← 最終的に更新
    vector3 *dVel,    // 引数② GPU側の速度配列（読み書き）← 最終的に更新
    vector3 *dAccels  // 引数③ GPU側の加速度行列（読み取り専用）
) {
    // ── 共有メモリ宣言 ────────────────────────────────
    __shared__ vector3 shAccel[256];
    // 役割 : Reduction用のバッファ
    // サイズ: 256 = blockDim.x（スレッド数と同じ）
    // 各スレッドが自分の部分和を書き込み
    // → Reductionで段階的に合計していく

    // ── インデックス ──────────────────────────────────
    int i   = blockIdx.x;
    // 物体のインデックス
    // ブロック数=NUMENTITIESなので
    // ブロック番号 = 物体番号 そのまま

    int tid = threadIdx.x;
    // ブロック内スレッド番号（0〜255）
    // どのj成分を担当するか決める
    // tid=0 → j=0, 256, 512...
    // tid=1 → j=1, 257, 513...

    if (i >= NUMENTITIES) return;

    // ── 部分和の計算 ──────────────────────────────────
    vector3 local;
    local[0] = 0;
    local[1] = 0;
    local[2] = 0;
    // このスレッドが担当するj成分の部分和
    // NUMENTITIESが256より大きいとき
    // 1スレッドが複数のjを担当する

    for (int j = tid; j < NUMENTITIES; j += blockDim.x) {
    //               ↑tidから始めて    ↑256刻みで進む
        for (int k = 0; k < 3; k++)
            local[k] += dAccels[i*NUMENTITIES + j][k];
    }

    for(int k=0;k<3;k++) shAccel[tid][k] = local[k];
    // 部分和を共有メモリに書き込む

    __syncthreads();
    // 全スレッドの書き込みを待つ

    // ── Reductionで合計 ───────────────────────────────
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //   stride: 128→64→32→16→8→4→2→1
    //   >>= 1 : 2で割る（ビットシフト）

        if (tid < stride) {
        // 前半スレッドだけが計算する
        // stride=4のとき: tid=0,1,2,3だけが動く
            for (int k = 0; k < 3; k++)
                shAccel[tid][k] += shAccel[tid + stride][k];
            // 前半が後半を足し込む
            // 例: shAccel[0] += shAccel[4]
            //     shAccel[1] += shAccel[5]  など
        }
        __syncthreads();
        // ↑各ステップごとに同期が必要！
        // これがないと次のステップで未完了の値を読む
    }

    // ── 速度・位置の更新 ──────────────────────────────
    if (tid == 0) {
    // shAccel[0]に全合計が入っているので
    // tid==0のスレッドだけが更新する
        for (int k = 0; k < 3; k++) {
            dVel[i][k] += shAccel[0][k] * INTERVAL;
            dPos[i][k] += dVel[i][k]    * INTERVAL;
        }
    }
}

// __global__ void updateBodies(
//     vector3 *dPos,    // 引数① GPU側の位置配列（読み書き）← 最終的に更新
//     vector3 *dVel,    // 引数② GPU側の速度配列（読み書き）← 最終的に更新
//     vector3 *dAccels  // 引数③ GPU側の加速度行列（読み取り専用）
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     // → 担当する物体のインデックス（1スレッド = 1物体）

//     if (i >= NUMENTITIES) return;

//     vector3 accel_sum = {0, 0, 0};
//     // 物体iへの合計加速度
//     // accel_sum[0] = x方向の合計
//     // accel_sum[1] = y方向の合計
//     // accel_sum[2] = z方向の合計

//     // i行目を全部足す（全物体からの影響を合計）
//     for (int j = 0; j < NUMENTITIES; j++) {
//         for (int k = 0; k < 3; k++)
//             accel_sum[k] += dAccels[i*NUMENTITIES + j][k];
//     }

//     // 速度・位置を更新
//     for (int k = 0; k < 3; k++) {
//         dVel[i][k] += accel_sum[k] * INTERVAL; // 速度 += 加速度 × 時間
//         dPos[i][k] += dVel[i][k]   * INTERVAL; // 位置 += 速度   × 時間
//     }
// }


//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){


	// 1: Take memory in GPU
	// 2: Transfer data from CPU to GPU
	// 3: compute Accels
	// 4: Update bodies(renew velocity and position)
	// 5: transfer result from GPU to CPU
	// 6: cudaFree


	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	// vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	// vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);

	vector3 *dPos; // GPUアドレスを入れる変数
	vector3 *dVel;
	double *dMass;
	vector3 *dAccels;

	cudaMalloc(&dPos, sizeof(vector3)*NUMENTITIES); // dPos(将来GPUアドレスが入る)自体のアドレスを渡す
	cudaMalloc(&dVel, sizeof(vector3)*NUMENTITIES);
	cudaMalloc(&dMass, sizeof(double)*NUMENTITIES);
	cudaMalloc(&dAccels, sizeof(vector3)*NUMENTITIES*NUMENTITIES);

	cudaMemcpy(

		dPos, //CPU
		hPos, //GPU
		sizeof(vector3)*NUMENTITIES, //size
		cudaMemcpyHostToDevice //CPU->GPU

	);

	cudaMemcpy(dVel,  hVel,  sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dMass, mass,  sizeof(double) *NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize(
		(NUMENTITIES + 15) / 16,
		(NUMENTITIES + 15) / 16
	);
	// computeAccels<<<gridSize, blockSize>>>(dPos, dMass, dAccels);
    computeAccels_shared<<<gridSize, blockSize>>>(dPos, dMass, dAccels);
    


	// for (i=0;i<NUMENTITIES;i++)
	// 	accels[i]=&values[i*NUMENTITIES];
	// //first compute the pairwise accelerations.  Effect is on the first argument.
	// for (i=0;i<NUMENTITIES;i++){
		// for (j=0;j<NUMENTITIES;j++){
			// if (i==j) {
			// 	FILL_VECTOR(accels[i][j],0,0,0);
			// }
			// else{
			// 	vector3 distance;
			// 	for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
			// 	double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
			// 	double magnitude=sqrt(magnitude_sq);
			// 	double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
			// 	FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			// }
	// 	}
	// }

	// dim3 blockSize2(256);
	// 1次元: 1ブロック256スレッド

	// dim3 gridSize2((NUMENTITIES + 255) / 256);
	// 必要なブロック数（切り上げ）
	// 例: NUMENTITIES=10  → 1ブロック
	//     NUMENTITIES=300 → 2ブロック

	// updateBodies<<<gridSize2, blockSize2>>>(dPos, dVel, dAccels);
	//                                       ↑更新先  ↑読み取り
    updateBodies_reduction<<<NUMENTITIES, 256>>>(dPos, dVel, dAccels);



	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	// for (i=0;i<NUMENTITIES;i++){
	// 	vector3 accel_sum={0,0,0};
	// 	for (j=0;j<NUMENTITIES;j++){
	// 		for (k=0;k<3;k++)
				// accel_sum[k]+=accels[i][j][k];
		// }
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		// for (k=0;k<3;k++){
		// 	hVel[i][k]+=accel_sum[k]*INTERVAL;
		// 	hPos[i][k]+=hVel[i][k]*INTERVAL;
		// }
	// }

	// GPU → CPU
	cudaMemcpy(hPos, dPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	// massは変わらないのでコピー不要

	// 解放
	cudaFree(dPos);
	cudaFree(dVel);
	cudaFree(dMass);
	cudaFree(dAccels);


	// free(accels);
	// free(values);
}
