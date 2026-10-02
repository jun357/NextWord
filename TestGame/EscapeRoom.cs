public sealed class EscapeRoom
{
    public GameState State { get; } = new();

    public string Execute(EscapeAction action)
    {
        if (State.Escaped)
            return "이미 방을 빠져나왔다.";

        return action switch
        {
            EscapeAction.LookAround
                => LookAround(),

            EscapeAction.InspectClock
                => InspectClock(),

            EscapeAction.InspectClockBack
                => InspectClockBack(),

            EscapeAction.ReadNote
                => ReadNote(),

            EscapeAction.InspectBookshelf
                => InspectBookshelf(),

            EscapeAction.InspectBook
                => InspectBook(),

            EscapeAction.OpenSafe
                => OpenSafe(),

            EscapeAction.TakeKey
                => TakeKey(),

            EscapeAction.OpenDrawer
                => OpenDrawer(),

            EscapeAction.InspectRadio
                => InspectRadio(),

            EscapeAction.TurnOnRadio
                => TurnOnRadio(),

            EscapeAction.InsertCassette
                => InsertCassette(),

            EscapeAction.PlayCassette
                => PlayCassette(),

            EscapeAction.Listen
                => Listen(),

            EscapeAction.InspectDoor
                => InspectDoor(),

            EscapeAction.OpenDoor
                => OpenDoor(),

            EscapeAction.Wait
                => Wait(),

            _ => "알 수 없는 행동이다."
        };
    }

    private string LookAround()
    {
        return
            "작은 방이다. " +
            "책장, 책상, 벽시계, 금고, 라디오와 출입문이 보인다.";
    }

    private string InspectClock()
    {
        State.ClockInspected = true;

        return
            "벽시계는 8시 17분에서 멈춰 있다.";
    }

    private string InspectClockBack()
    {
        if (!State.ClockInspected)
            return "시계를 먼저 살펴보는 것이 좋겠다.";

        State.NoteFound = true;

        return
            "시계 뒤에서 작은 쪽지를 발견했다.";
    }

    private string ReadNote()
    {
        if (!State.NoteFound)
            return "읽을 쪽지가 없다.";

        State.NoteRead = true;

        return
            "쪽지에는 이렇게 적혀 있다.\n" +
            "\"시간은 거꾸로 기억한다.\"";
    }

    private string InspectBookshelf()
    {
        State.BookshelfInspected = true;

        return
            "책장에는 오래된 책 네 권이 있다. " +
            "책등에 희미한 숫자가 보인다.";
    }

    private string InspectBook()
    {
        if (!State.BookshelfInspected)
            return "책장을 먼저 살펴보는 것이 좋겠다.";

        State.BookInspected = true;

        return
            "책등에서 3, 8, 1, 6이라는 숫자를 발견했다.";
    }

    private string OpenSafe()
    {
        if (!State.BookInspected)
            return "금고를 열 수 있는 단서를 아직 찾지 못했다.";

        if (!State.NoteRead)
            return "책에서 발견한 숫자의 의미를 아직 알 수 없다.";

        State.SafeOpened = true;
        State.KeyObtained = true;
        State.CassetteFound = true;

        return
            "금고가 열렸다. " +
            "안에서 작은 열쇠와 카세트테이프를 발견했다.";
    }

    private string TakeKey()
    {
        if (!State.SafeOpened)
            return "열쇠가 보이지 않는다.";

        State.KeyObtained = true;

        return "작은 열쇠를 집었다.";
    }

    private string OpenDrawer()
    {
        if (!State.KeyObtained)
            return "잠긴 서랍이다. 열쇠가 필요하다.";

        State.DrawerOpened = true;
        State.RadioFound = true;

        return
            "서랍이 열렸다. " +
            "안에서 오래된 라디오를 발견했다.";
    }

    private string InspectRadio()
    {
        if (!State.RadioFound)
            return "라디오가 보이지 않는다.";

        return
            "오래된 라디오다. " +
            "카세트 플레이어가 붙어 있다.";
    }

    private string TurnOnRadio()
    {
        if (!State.RadioFound)
            return "라디오가 없다.";

        State.RadioOn = true;

        return
            "라디오를 켰다. 지직거리는 잡음이 흘러나온다.";
    }

    private string InsertCassette()
    {
        if (!State.CassetteFound)
            return "카세트테이프를 찾지 못했다.";

        if (!State.RadioFound)
            return "카세트를 넣을 장치가 없다.";

        State.CassetteInserted = true;

        return "카세트테이프를 라디오에 넣었다.";
    }

    private string PlayCassette()
    {
        if (!State.CassetteInserted)
            return "카세트가 들어 있지 않다.";

        State.CassettePlayed = true;

        return "카세트가 재생되기 시작한다.";
    }

    private string Listen()
    {
        if (!State.RadioFound || !State.RadioOn)
            return "고요하다...";

        if (!State.CassetteInserted)
            return "지직... 약한 잡음만 들린다.";

        if (!State.CassettePlayed)
            return "카세트가 들어 있지만 아무 소리도 재생되지 않는다.";

        return
            "삐.\n" +
            "삐삐.\n" +
            "삐삐삐삐.";
    }

    private string InspectDoor()
    {
        return
            "출입문은 잠겨 있다. " +
            "문 옆에는 작은 숫자 입력 장치가 있다.";
    }

    private string OpenDoor()
    {
        if (!State.CassettePlayed)
            return "문을 열 수 없다. 아직 탈출 방법을 찾지 못했다.";

        State.DoorUnlocked = true;
        State.Escaped = true;

        return
            "철컥—\n" +
            "문이 열렸다.\n" +
            "탈출 성공!";
    }

    private string Wait()
    {
        return "시간이 흐른다... 특별한 변화는 없다.";
    }
}
