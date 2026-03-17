/*
 * utlz.c
 *
 *  Created on: Sep 9, 2024
 *      Author: Chaoxiang Ye

 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "utlz.h"
#include "physicalConstants.h"
#include "debug.h"
#include "GPIS.h"



float calculate_rotation_time(float p_x, float p_y, float max_x, float max_y, float current_yaw, float max_turn_rate, float* rotation_direction) 
{
    // Compute the vector from the current position to the target point
    float delta_x = max_x - p_x;
    float delta_y = max_y - p_y;

    // Compute the target heading angle with respect to the x-axis, in degrees
    float target_angle = (float)atan2(delta_y, delta_x) * (180.0f / M_PI_F);

    // Compute the heading difference between the current yaw and the target angle
    float angle_diff = target_angle - current_yaw;

    // Wrap
    while (angle_diff > 180.0f) angle_diff -= 360.0f;
    while (angle_diff < -180.0f) angle_diff += 360.0f;

    // Determine the rotation direction
    float rotation_time = 0.0f;
    if (angle_diff < 0) 
    {
        *rotation_direction = -1.0f; // Clockwise rotation
    }
    else
    {
        *rotation_direction = 1.0f;
    }
    rotation_time = 50.0f * fabsf(angle_diff) /max_turn_rate;

    DEBUG_PRINT("Target Angle: %f degrees, Current Yaw: %f degrees\n", (double)target_angle, (double)current_yaw);
    DEBUG_PRINT("Angle difference: %f degrees. Rotation time: %f\n", (double)angle_diff, (double)rotation_time);

    return rotation_time; 
}

// Compute the Euclidean distance between two 2D points
float euclidean_distance(float *p1, float *p2) 
{
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}


void compute_normal(Point *p1, Point *p2, float *normal) 
{
    // Compute the unnormalized normal vector of the segment p1 -> p2
    float dx = p2->x - p1->x;
    float dy = p2->y - p1->y;

    // Use one of the two perpendicular directions
    normal[0] = dy;
    normal[1] = -dx;

    // Normalize the normal vector
    float length = sqrt(normal[0] * normal[0] + normal[1] * normal[1]);
    if (length != 0) {
        normal[0] /= length;
        normal[1] /= length;
    }
}


// Compute the dot product between two 2D vectors
float dot_product(float *v1, float *v2) {
    return v1[0] * v2[0] + v1[1] * v2[1];
}

// Compute the centroid of a cluster of 2D points
void compute_centroid(float *cluster, int cluster_size, float *centroid) 
{
    centroid[0] = 0;
    centroid[1] = 0;
    for (int i = 0; i < cluster_size; ++i) {
        centroid[0] += cluster[i * 2];
        centroid[1] += cluster[i * 2 + 1];
    }
    centroid[0] /= cluster_size;
    centroid[1] /= cluster_size;
}

// void find_high_curvature_clusters_with_normals(Point *points, int num_points, float max_normal_threshold, float min_normal_threshold, float *significant_points, int *num_significant_points) {
//     float current_cluster[50]; // Stores the current cluster as a flattened list of x-y pairs
//     int cluster_size = 0; 

//     *num_significant_points = 0; // Initialize the number of detected significant points

//     for (int i = 1; i < num_points; ++i) 
//     {
//         Point *p1, *p2, *p3;
//         if (i < num_points - 1) {
//             p1 = &points[i - 1]; 
//             p2 = &points[i];     
//             p3 = &points[i + 1];  
//         } else {
//             p1 = &points[i - 1]; 
//             p2 = &points[i];      
//             p3 = &points[0];  
//         }


//         float normal1[2], normal2[2];
//         compute_normal(p1, p2, normal1);
//         compute_normal(p2, p3, normal2);

//         // Measure the similarity between the two normals
//         float cos_theta = dot_product(normal1, normal2);

//         // If the normal change is large, close the current cluster and mark the point
//         if (cos_theta < min_normal_threshold) {
//             // add to cluster
//             if (cluster_size > 0) { 
//                 float centroid[2];
//                 compute_centroid(current_cluster, cluster_size, centroid);

//                 significant_points[(*num_significant_points) * 2] = centroid[0];
//                 significant_points[(*num_significant_points) * 2 + 1] = centroid[1];
//                 (*num_significant_points)++;
//             significant_points[(*num_significant_points) * 2] = p2->x;
//             significant_points[(*num_significant_points) * 2 + 1] = p2->y;
//             cluster_size = 0; // reset
//             }
//         } 
//         else if (cos_theta < max_normal_threshold) 
//         {
//             if (cluster_size < 25) { 
//                 current_cluster[cluster_size * 2] = p2->x;
//                 current_cluster[cluster_size * 2 + 1] = p2->y;
//                 cluster_size++;
//             }
//         }
//         else {
//             // End the current cluster if the normals are too similar
//             if (cluster_size > 0) {
//                 float centroid[2];
//                 compute_centroid(current_cluster, cluster_size, centroid);

//                 significant_points[(*num_significant_points) * 2] = centroid[0];
//                 significant_points[(*num_significant_points) * 2 + 1] = centroid[1];
//                 (*num_significant_points)++;

//                 cluster_size = 0; 
//             }
//         }
//     }

//     // Finalize the last cluster if any points remain
//     if (cluster_size > 0) {
//         float centroid[2];
//         compute_centroid(current_cluster, cluster_size, centroid);

//         significant_points[(*num_significant_points) * 2] = centroid[0];
//         significant_points[(*num_significant_points) * 2 + 1] = centroid[1];
//         (*num_significant_points)++;
//     }
    
// }

// penalty function
float calculate_penalty(float *point, float *significant_points, int num_significant_points, float c) 
{
    float min_distance = FLT_MAX;

    for (int i = 0; i < num_significant_points; ++i) 
    {
        float *sig_point = &significant_points[i * 2];
        float distance = euclidean_distance(point, sig_point);
        if (distance < min_distance) {
            min_distance = distance;
        }
    }

    return -exp(-min_distance * min_distance / (2 * c * c));
}


void apply_penalty(Point *contour_points, int num_contour_points, float *y_stds, float *significant_points, int num_significant_points, float c) 
{
    for (int i = 0; i < num_contour_points; ++i) 
    {
        
        float current_point[2] = { contour_points[i].x, contour_points[i].y }; 
        float penalty = calculate_penalty(current_point, significant_points, num_significant_points, c);
        contour_points[i].y_std += penalty; 
        DEBUG_PRINT("orderedContourPoints with penalty x: %f y: %f std: %f\n", (double)contour_points[i].x, (double)contour_points[i].y,(double)contour_points[i].y_std);
    }
}


void saveLineSegment(LineSegment *lineSegments, float x1, float y1, float y_std1, float x2, float y2, float y_std2, int *segmentCount) 
{
    lineSegments[*segmentCount] = (LineSegment){{x1, y1, y_std1}, {x2, y2, y_std2}};
    (*segmentCount)++;
}


void saveOrderedContourPoint(Point *orderedContourPoints, float x, float y, float y_std, int *orderedPointCount) 
{
    orderedContourPoints[*orderedPointCount] = (Point){x, y, y_std};
    DEBUG_PRINT("orderedContourPoints x: %f y: %f std: %f\n", (double)x, (double)y,(double)y_std);
    (*orderedPointCount)++;
}

void marchingSquares(int grid_size, float *y_preds, float *y_stds, float x_min, float x_step, float y_min, float y_step, LineSegment *lineSegments, int *segmentCount)  
{
    for (int i = 0; i < grid_size - 1; i++) 
    {
        for (int j = 0; j < grid_size - 1; j++) 
        {
            int idx_00 = i * grid_size + j;    
            int idx_01 = i * grid_size + (j + 1); 
            int idx_10 = (i + 1) * grid_size + j; 
            int idx_11 = (i + 1) * grid_size + (j + 1); 

            
            int topLeft     = y_preds[idx_00] > 0;
            int topRight    = y_preds[idx_01] > 0;
            int bottomLeft  = y_preds[idx_10] > 0;
            int bottomRight = y_preds[idx_11] > 0;

            
            int cellIndex = (topLeft << 3) | (topRight << 2) | (bottomRight << 1) | bottomLeft;

            
            switch (cellIndex) 
            {
                case 1: // 0001
                case 14: // 1110
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_10]));
                        float x1 = x_min + j * x_step;
                        float y1 = y_min + i * y_step + t1 * y_step;
                        float y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_10] - y_stds[idx_00]);

                        float t2 = fabsf(y_preds[idx_10]) / (fabsf(y_preds[idx_10]) + fabsf(y_preds[idx_11]));
                        float x2 = x_min + j * x_step + t2 * x_step;
                        float y2 = y_min + (i + 1) * y_step;
                        float y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2, segmentCount);
                    }
                    break;
                case 2: // 0010
                case 13: // 1101
                    {
                        float t1 = fabsf(y_preds[idx_01]) / (fabsf(y_preds[idx_01]) + fabsf(y_preds[idx_11]));
                        float x1 = x_min + (j + 1) * x_step;
                        float y1 = y_min + i * y_step + t1 * y_step;
                        float y_std1 = y_stds[idx_01] + t1 * (y_stds[idx_11] - y_stds[idx_01]);

                        float t2 = fabsf(y_preds[idx_10]) / (fabsf(y_preds[idx_10]) + fabsf(y_preds[idx_11]));
                        float x2 = x_min + j * x_step + t2 * x_step;
                        float y2 = y_min + (i + 1) * y_step;
                        float y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2, segmentCount);
                    }
                    break;
                case 3: // 0011
                case 12: // 1100
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_10]));
                        float x1 = x_min + j * x_step;
                        float y1 = y_min + i * y_step + t1 * y_step;
                        float y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_10] - y_stds[idx_00]);

                        float t2 = fabsf(y_preds[idx_01]) / (fabsf(y_preds[idx_01]) + fabsf(y_preds[idx_11]));
                        float x2 = x_min + (j + 1) * x_step;
                        float y2 = y_min + i * y_step + t2 * y_step;
                        float y_std2 = y_stds[idx_01] + t2 * (y_stds[idx_11] - y_stds[idx_01]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2, segmentCount);
                    }
                    break;
                case 4: // 0100
                case 11: // 1011
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_01]));
                        float x1 = x_min + j * x_step + t1 * x_step;
                        float y1 = y_min + i * y_step;
                        float y_std1 = y_stds[idx_01] + t1 * (y_stds[idx_01] - y_stds[idx_00]);

                        float t2 = fabsf(y_preds[idx_01]) / (fabsf(y_preds[idx_01]) + fabsf(y_preds[idx_11]));
                        float x2 = x_min + (j + 1) * x_step;
                        float y2 = y_min + i * y_step + t2 * y_step;
                        float y_std2 = y_stds[idx_01] + t2 * (y_stds[idx_11] - y_stds[idx_01]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2, segmentCount);
                    }
                    break;
                case 5: // 0101
                case 10:
                    break;
                case 6: // 0110
                case 9:
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_01]));
                        float x1 = x_min + j * x_step + t1 * x_step;
                        float y1 = y_min + i * y_step;
                        float y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_01] - y_stds[idx_00]);

                        float t2 = fabsf(y_preds[idx_10]) / (fabsf(y_preds[idx_11]) + fabsf(y_preds[idx_10]));
                        float x2 = x_min + j * x_step + t2 * x_step;
                        float y2 = y_min + (i + 1) * y_step;
                        float y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2, segmentCount);
                    }
                    break;
                case 7: // 0111
                case 8: // 1000
                    {
                        float t1 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_01]) + fabsf(y_preds[idx_00]));
                        float x1 = x_min + j * x_step + t1 * x_step;
                        float y1 = y_min + i * y_step;
                        float y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_00] - y_stds[idx_10]);

                        float t2 = fabsf(y_preds[idx_00]) / (fabsf(y_preds[idx_00]) + fabsf(y_preds[idx_10]));
                        float x2 = x_min + j * x_step;
                        float y2 = y_min + i * y_step + t2 * y_step;
                        float y_std2 = y_stds[idx_00] + t2 * (y_stds[idx_10] - y_stds[idx_00]);

                        saveLineSegment(lineSegments, x1, y1, y_std1, x2, y2, y_std2, segmentCount);
                    }
                    break;
                    
                case 0: //0000
                case 15: // 1111
                    
                    break;
                default:
                    
                    break;
            }
        }
    }
}


void connectContourSegments(LineSegment *lineSegments, Point *orderedContourPoints, int *segmentCount, int *orderedPointCount) 
{
    int visited[30] = {0};
    int foundStart = 0;

   
    for (int i = 0; i < *segmentCount; i++) {
        if (visited[i]) continue;

        
        saveOrderedContourPoint(orderedContourPoints, lineSegments[i].start.x, lineSegments[i].start.y, lineSegments[i].start.y_std, orderedPointCount);
        saveOrderedContourPoint(orderedContourPoints, lineSegments[i].end.x, lineSegments[i].end.y, lineSegments[i].end.y_std, orderedPointCount);
        visited[i] = 1;

        Point currentEnd = lineSegments[i].end;
        foundStart = 1;

        
        while (foundStart) {
            foundStart = 0;
            for (int j = 0; j < *segmentCount; j++) {
                if (!visited[j]) {
                    if (fabs(lineSegments[j].start.x - currentEnd.x) < 1e-6 &&
                        fabs(lineSegments[j].start.y - currentEnd.y) < 1e-6) {
                       
                        saveOrderedContourPoint(orderedContourPoints, lineSegments[j].end.x, lineSegments[j].end.y, lineSegments[j].end.y_std, orderedPointCount);
                        currentEnd = lineSegments[j].end;
                        visited[j] = 1;
                        foundStart = 1;
                        break;
                    }
                    else if (fabs(lineSegments[j].end.x - currentEnd.x) < 1e-6 &&
                             fabs(lineSegments[j].end.y - currentEnd.y) < 1e-6) {
                        
                        saveOrderedContourPoint(orderedContourPoints, lineSegments[j].start.x, lineSegments[j].start.y, lineSegments[j].start.y_std, orderedPointCount);
                        currentEnd = lineSegments[j].start;
                        visited[j] = 1;
                        foundStart = 1;
                        break;
                    }
                }
            }
        }
    }
}


void compute_curvature_kernel(const GaussianProcess *gp,
    Point *contour_points, int contour_count,
    float *curvatures
) {
    
    float weights[gp->train_size];
    for (int i = 0; i < gp->train_size; ++i) {
        weights[i] = 0.0f;
        for (int j = 0; j < gp->train_size; ++j) {
            weights[i] += gp->K_inv[i * gp->train_size + j] * gp->y_train[j];
        }
    }
    for (int i = 0; i < contour_count; i++) {
        Point x = contour_points[i];

        float grad[2] = {0};
        float hessian[2][2] = {{0, 0}, {0, 0}};
        float grad_norm = 0;

        for (int j = 0; j < gp->train_size; j++) {
            float xi_x = gp->X_train[j * 2];
            float xi_y = gp->X_train[j * 2 + 1];
            float w = weights[j];

            float diff_x = x.x - xi_x;
            float diff_y = x.y - xi_y;
            float d2 = diff_x * diff_x + diff_y * diff_y;
            float d2_c2 = d2 + 1.0f;

            
            float coeff = -w / powf(d2_c2, 1.5f);
            grad[0] += coeff * diff_x;
            grad[1] += coeff * diff_y;

            
            float outer_00 = diff_x * diff_x;
            float outer_01 = diff_x * diff_y;
            float outer_11 = diff_y * diff_y;
            float hess_coeff1 = 3 * w / powf(d2_c2, 2.5f);
            float hess_coeff2 = w / powf(d2_c2, 1.5f);

            hessian[0][0] += hess_coeff1 * outer_00 - hess_coeff2;
            hessian[0][1] += hess_coeff1 * outer_01;
            hessian[1][0] += hess_coeff1 * outer_01;
            hessian[1][1] += hess_coeff1 * outer_11 - hess_coeff2;
        }

        
        grad_norm = sqrtf(grad[0] * grad[0] + grad[1] * grad[1]);
        if (grad_norm == 0) {
            curvatures[i] = 0;
            continue;
        }

        
        float tr_hessian = hessian[0][0] + hessian[1][1];
        float grad_hess_grad = 
            grad[0] * (hessian[0][0] * grad[0] + hessian[0][1] * grad[1]) +
            grad[1] * (hessian[1][0] * grad[0] + hessian[1][1] * grad[1]);
        curvatures[i] = (grad_hess_grad - grad_norm * grad_norm * tr_hessian) / 
                        (grad_norm * grad_norm * grad_norm);
    }
}

void find_high_curvature_clusters_using_curvature(
    Point *contour_points, int num_points, 
    float *curvatures, float curvature_threshold, 
    float *significant_points, int *num_significant_points
) {
    float current_cluster[50]; 
    int cluster_size = 0;    

    *num_significant_points = 0; 

    float first_centroid[2] = {0}; 
    float last_centroid[2] = {0};  
    int first_centroid_calculated = 0; 
    int last_centroid_calculated = 0;  

    for (int i = 0; i < num_points; ++i) {
        if (curvatures[i] < curvature_threshold) {
            
            if (cluster_size < 25) { 
                current_cluster[cluster_size * 2] = contour_points[i].x;
                current_cluster[cluster_size * 2 + 1] = contour_points[i].y;
                cluster_size++;
            }
        } else {
            
            if (cluster_size > 0) {
                float centroid[2];
                compute_centroid(current_cluster, cluster_size, centroid);

                
                if (!first_centroid_calculated) {
                    first_centroid[0] = centroid[0];
                    first_centroid[1] = centroid[1];
                    first_centroid_calculated = 1;
                }

                
                last_centroid[0] = centroid[0];
                last_centroid[1] = centroid[1];
                last_centroid_calculated = 1;

                
                significant_points[(*num_significant_points) * 2] = centroid[0];
                significant_points[(*num_significant_points) * 2 + 1] = centroid[1];
                (*num_significant_points)++;

                cluster_size = 0; 
            }
        }
    }


    if (cluster_size > 0) {
        float centroid[2];
        compute_centroid(current_cluster, cluster_size, centroid);

        if (!first_centroid_calculated) {
            first_centroid[0] = centroid[0];
            first_centroid[1] = centroid[1];
            first_centroid_calculated = 1;
        }

        last_centroid[0] = centroid[0];
        last_centroid[1] = centroid[1];
        last_centroid_calculated = 1;

        significant_points[(*num_significant_points) * 2] = centroid[0];
        significant_points[(*num_significant_points) * 2 + 1] = centroid[1];
        (*num_significant_points)++;
    }

    
    if (first_centroid_calculated && last_centroid_calculated &&
        curvatures[0] < curvature_threshold && curvatures[num_points - 1] < curvature_threshold) {
        
        
        float merged_centroid[2];
        merged_centroid[0] = (first_centroid[0] + last_centroid[0]) / 2.0f;
        merged_centroid[1] = (first_centroid[1] + last_centroid[1]) / 2.0f;

    
        significant_points[(*num_significant_points) * 2] = merged_centroid[0];
        significant_points[(*num_significant_points) * 2 + 1] = merged_centroid[1];
        (*num_significant_points)++;
    }
}
