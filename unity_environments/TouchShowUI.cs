using UnityEngine;
using UnityEngine.EventSystems;

public class TouchShowUI : MonoBehaviour, IPointerDownHandler
{
    public GameObject buttonPanel;
    bool shown = false;

    void Start()
    {
        buttonPanel.SetActive(false);
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        if (Input.touchCount <= 0) return;

        if (shown) return;

        buttonPanel.SetActive(true);
        shown = true;
    }
}
