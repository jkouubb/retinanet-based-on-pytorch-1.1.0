from data_loader import CSV_Dataset, Normalizer, Resizer, AspectRatioBasedSampler, collater
from torchvision import transforms
from Build_Network import RetinaNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from others import compute_overlap, _compute_ap

def run_the_net(train_file, class_list, test_file=None):
    print('loading data')
    dataset_train = CSV_Dataset(train_file, class_list, transform=transforms.Compose([Normalizer(),  Resizer()]))
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    print('initializing retinanet')
    model = RetinaNet(dataset_train.num_classes())
    if torch.cuda.is_available():
        model = model.cuda()
    print('initializing paremeters')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.training = True
    print('training')
    for i in range(100):
        model.train()
        average_loss = 0.0

        for data in dataloader_train:
            classification_loss, regression_loss = model([data['image'].cuda().float(), data['annotations']])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            average_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        print('the average loss in no.%d training:' % (i+1), average_loss / dataset_train.__len__().__float__())
    if test_file is None:
        print('test file not given')
        return
    print('loading test data')
    dataset_test = CSV_Dataset(test_file, class_list, transform=transforms.Compose([Normalizer(), Resizer()]))

    print('initializing paremeters')
    scores_threshold = 0.05
    max_boxes_per_image = 100
    iou_threshold = 0.5

    model.training = False
    model.eval()

    print('testing')
    all_detections = [[None for i in range(dataset_test.num_classes())] for j in range(len(dataset_test))]
    all_annotations = [[None for i in range(dataset_test.num_classes())] for j in range(len(dataset_test))]
    for image_id, data in enumerate(dataset_test):
        """
        part 1:
            get detections and adjust them
        """
        scores, classes, boxes = model(data['image'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
        scale = data['scale']

        scores = scores.cpu().detach().numpy()
        classes = classes.cpu().detach().numpy()
        boxes = boxes.cpu().detach().numpy()
        boxes = boxes / scale

        valid_scores_indices = np.where(scores > scores_threshold)[0]
        if valid_scores_indices.shape[0] > 0:
            scores = scores[valid_scores_indices]

            scores_sort = np.argsort(-scores)[:max_boxes_per_image]

            # select detections
            image_boxes = boxes[valid_scores_indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_labels = classes[valid_scores_indices[scores_sort]]
            image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
            for label in range(dataset_test.num_classes()):
                all_detections[image_id][label] = image_detections[image_detections[:, -1] == label, :-1]
        else:
        # copy detections to all_detections
            for label in range(dataset_test.num_classes()):
                all_detections[image_id][label] = np.zeros((0, 5))


        """
        part 2:
            get annotations
        """
        annotation = dataset_test.read_annotations(image_id)
        for label in range(dataset_test.num_classes()):
            all_annotations[image_id][label] = annotation[annotation[:, 4] == label, :4].copy()


    """
    part 3:
    calculate mAP
    """
    average_precisions = {}

    for label in range(dataset_test.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(dataset_test)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print('\nmAP:')
    for label in range(dataset_test.num_classes()):
        label_name = dataset_test.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))


if __name__ == '__main__':
    run_the_net('data/train/annotations.csv', 'data/train/classes.csv', 'data/train/annotations.csv')
