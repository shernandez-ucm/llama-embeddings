#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <limits>

/**
 * @brief A simple KD-Tree implementation for nearest neighbor search using Eigen.
 * @tparam Scalar The scalar type (e.g., float, double).
 * @tparam Dim The dimensionality of the vectors.
 */
template <typename Scalar, int Dim>
class KDTree {
public:
    using Vector = Eigen::Matrix<Scalar, Dim, 1>;

    struct Node {
        Vector point;
        size_t index;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node(const Vector& p, size_t idx) : point(p), index(idx), left(nullptr), right(nullptr) {}
    };

    /**
     * @brief Build the KD-Tree from a list of points.
     * @param points A vector of Eigen vectors.
     */
    KDTree(const std::vector<Vector>& points) {
        std::vector<size_t> indices(points.size());
        std::iota(indices.begin(), indices.end(), 0);
        root_ = buildRecursive(points, indices.begin(), indices.end(), 0);
    }

    /**
     * @brief Find the nearest neighbor to a query point.
     * @param query The query vector.
     * @param out_index Output parameter for the index of the nearest neighbor.
     * @param out_distance Output parameter for the squared distance.
     */
    void nearestNeighbor(const Vector& query, size_t& out_index, Scalar& out_distance) const {
        out_distance = std::numeric_limits<Scalar>::max();
        searchRecursive(root_.get(), query, 0, out_index, out_distance);
    }

private:
    std::unique_ptr<Node> root_;

    using IndexIterator = typename std::vector<size_t>::iterator;

    std::unique_ptr<Node> buildRecursive(const std::vector<Vector>& points, IndexIterator begin, IndexIterator end, int depth) {
        if (begin == end) return nullptr;

        int axis = depth % Dim;
        IndexIterator mid = begin + std::distance(begin, end) / 2;

        std::nth_element(begin, mid, end,
                         [&](size_t a, size_t b) {
                             return points[a](axis) < points[b](axis);
                         });

        auto node = std::make_unique<Node>(points[*mid], *mid);
        
        node->left = buildRecursive(points, begin, mid, depth + 1);
        node->right = buildRecursive(points, mid + 1, end, depth + 1);

        return node;
    }

    void searchRecursive(const Node* node, const Vector& query, int depth, size_t& best_idx, Scalar& best_dist_sq) const {
        if (!node) return;

        Scalar dist_sq = (node->point - query).squaredNorm();
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = node->index;
        }

        int axis = depth % Dim;
        Scalar diff = query(axis) - node->point(axis);

        Node* near = diff < 0 ? node->left.get() : node->right.get();
        Node* far = diff < 0 ? node->right.get() : node->left.get();

        searchRecursive(near, query, depth + 1, best_idx, best_dist_sq);

        // Check if we need to search the other side
        if (diff * diff < best_dist_sq) {
            searchRecursive(far, query, depth + 1, best_idx, best_dist_sq);
        }
    }
};

#endif // KDTREE_HPP
