#ifndef UTLZ_H
#define UTLZ_H
#include "GPIS.h"

typedef struct {
    float x, y;    
    float y_std;     
} Point;

typedef struct {
    Point start;    
    Point end;    
} LineSegment;

extern int segmentCount;          
extern int orderedPointCount;    

void saveLineSegment(LineSegment *lineSegments, float x1, float y1, float y_std1, float x2, float y2, float y_std2, int *segmentCount);
void saveOrderedContourPoint(Point *orderedContourPoints, float x, float y, float y_std, int *orderedPointCount);
void marchingSquares(int grid_size, float *y_preds, float *y_stds, float x_min, float x_step, float y_min, float y_step, LineSegment *lineSegments, int *segmentCount);
void connectContourSegments(LineSegment *lineSegments, Point *orderedContourPoints, int *segmentCount, int *orderedPointCount);


void find_high_curvature_clusters_with_normals(Point *points, int num_points, float max_normal_threshold, float min_normal_threshold, float *significant_points, int *num_significant_points);


float calculate_rotation_time(float p_x, float p_y, float max_x, float max_y, float current_yaw, float max_turn_rate, float* rotation_direction);


void apply_penalty(Point *contour_points, int num_contour_points, float *y_stds, float *significant_points, int num_significant_points, float c);

void compute_curvature_kernel(const GaussianProcess *gp,
    Point *contour_points, int contour_count,
    float *curvatures
);

void find_high_curvature_clusters_using_curvature(
    Point *contour_points, int num_points, 
    float *curvatures, float curvature_threshold, 
    float *significant_points, int *num_significant_points
);
#endif // UTLZ_H